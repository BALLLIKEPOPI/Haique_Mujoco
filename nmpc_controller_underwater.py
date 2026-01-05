# 四旋翼NMPC控制器 20250304 Wakkk
# ACADOS NMPC
from acados_template import AcadosOcp, AcadosOcpSolver
from export_model import *
from export_model_underwater import *
import numpy as np
import scipy.linalg
from os.path import dirname, join, abspath
import time
from typing import Optional

from config_loader import get_mode_config, get_value

# np.set_printoptions(precision=3)  # 设置精度
np.set_printoptions(suppress=True)  # 禁用科学计数法输出

# ACADOS NMPC控制器
class NMPC_Controller:
    def __init__(
        self,
        left_servo_offset: Optional[float] = None,
        right_servo_offset: Optional[float] = None,
        config_path: str = join(dirname(abspath(__file__)), "config.yaml"),
    ):
        self.ocp = AcadosOcp()       # OCP 优化问题

        cfg = get_mode_config("underwater", path=config_path)

        # --- physical/model/servo params (single source of truth) ---
        self.g0 = float(get_value(cfg, "physical.g", 9.8066))
        self.mq = float(get_value(cfg, "physical.mass", 4.672))
        inertia = get_value(cfg, "physical.inertia", [0.10170715, 0.10222875, 0.16095642])
        self.inertia = np.asarray(inertia, dtype=float)
        self.Ct = float(get_value(cfg, "physical.Ct", 0.0267))
        self.Cd = float(get_value(cfg, "physical.Cd", 0.00111))
        self.dq = float(get_value(cfg, "physical.dq", 0.605))
        self.k_yaw_lr = float(get_value(cfg, "model.k_yaw_lr", 1.5))
        self.k_yaw_other = float(get_value(cfg, "model.k_yaw_other", 0.3))

        cfg_left_offset = float(get_value(cfg, "servo.left_offset", -0.1))
        cfg_right_offset = float(get_value(cfg, "servo.right_offset", 0.0))
        self.left_servo_offset = float(cfg_left_offset if left_servo_offset is None else left_servo_offset)
        self.right_servo_offset = float(cfg_right_offset if right_servo_offset is None else right_servo_offset)

        # self.model = export_model()  # 导出四旋翼物理模型
        self.model = export_model_underwater(
            g0=self.g0,
            mass=self.mq,
            inertia=tuple(float(x) for x in self.inertia.tolist()),
            Ct=self.Ct,
            Cd=self.Cd,
            dq=self.dq,
            alpha_offset=self.left_servo_offset,
            beta_offset=self.right_servo_offset,
            k_yaw_lr=self.k_yaw_lr,
            k_yaw_other=self.k_yaw_other,
        )
        
        # 数据记录
        self.data_log = []

        self.Tf = float(get_value(cfg, "nmpc.Tf", 0.5))          # 预测时间长度(s)
        self.N = int(get_value(cfg, "nmpc.N", 30))              # 预测步数(节点数量)
        self.nx = self.model.x.size()[0]    # 状态维度 13维度
        self.nu = self.model.u.size()[0]    # 控制输入维度 10维度（8电机+2舵机）
        self.ny = self.nx + self.nu         # 评估维度
        self.ny_e = self.nx                 # 终端评估维度

        # set ocp_nlp_dimensions
        self.nlp_dims     = self.ocp.dims
        self.nlp_dims.N   = self.N

        # bounds
        # 悬停（仅 1/3/5/7 四个竖直推进器承担重力）：mq*g ≈ 4*Ct*hov_w^2
        self.hov_w = np.sqrt((self.mq * self.g0) / (4 * self.Ct))
        print(f"hovor speed: {self.hov_w} krpm")
        # hovor speed: ~20.7 krpm (with Ct=0.0267)

        self.max_speed = float(get_value(cfg, "nmpc.max_speed", 30.0))  # 电机最高转速(krpm)

        # 舵机几何偏好：尽量与机体 x 轴平行（物理角 alpha/beta ≈ pi/2），允许小幅差动来生成 z 分力并平衡 roll。
        # Mujoco 中实际舵机角 = 控制指令 + offset；因此让“中心值”落在指令空间：cmd_center = pi/2 - offset。
        self.servo_center_alpha = (np.pi / 2) - self.left_servo_offset
        self.servo_center_beta = (np.pi / 2) - self.right_servo_offset
        # 允许偏转范围：增加余量以提升抗扰（代价会把舵机拉回中心，只有需要时才会偏转）
        self.servo_max_dev = np.deg2rad(float(get_value(cfg, "servo.max_dev_deg", 45.0)))

        # 扰动参数预处理：ESO 输出可能存在尖峰/尺度失真，直接送入 NMPC 会导致控制饱和与不收敛。
        # 这里做最小化处理：逐轴限幅 + 一阶低通（不改变接口，不依赖 ESO）。
        self.dist_force_clip = np.asarray(get_value(cfg, "nmpc.dist_force_clip", [60.0, 60.0, 60.0]), dtype=float)    # N
        self.dist_torque_clip = np.asarray(get_value(cfg, "nmpc.dist_torque_clip", [12.0, 12.0, 12.0]), dtype=float)  # Nm
        self.dist_lpf_alpha = float(get_value(cfg, "nmpc.dist_lpf_alpha", 0.20))  # 0~1，越大越“跟随”新估计
        self._disturbance_filt = np.zeros(6)

        # set weighting matrices 状态权重矩阵
        q_diag = get_value(cfg, "nmpc.Q_diag", None)
        if isinstance(q_diag, list) and len(q_diag) == self.nx:
            Q = np.diag(np.asarray(q_diag, dtype=float))
        else:
            Q = np.eye(self.nx)
            # 抗外部干扰：提高位置与速度阻尼权重，避免在持续扰动下“越飘越远”。
            Q[0,0] = 300.0       # x
            Q[1,1] = 300.0       # y
            Q[2,2] = 320.0       # z
            # 姿态权重 (约束飞行器保持水平姿态) - 极其重要！
            Q[3,3] = 0.0         # qw (标量部分，通常不直接约束)
            Q[4,4] = 120         # qx (roll)
            Q[5,5] = 360         # qy (pitch)
            Q[6,6] = 50.0        # qz (yaw)
            Q[7,7] = 120.0       # vx
            Q[8,8] = 120.0       # vy
            Q[9,9] = 90.0        # vz
            Q[10,10] = 120.0     # wx
            Q[11,11] = 120.0     # wy
            Q[12,12] = 80.0      # wz

        r_diag = get_value(cfg, "nmpc.R_diag", None)
        if isinstance(r_diag, list) and len(r_diag) == self.nu:
            R = np.diag(np.asarray(r_diag, dtype=float))
        else:
            R = np.eye(self.nu)   # 控制输入权重矩阵
            # 持续扰动下需要更“敢用力”，适当降低电机代价。
            for i in range(8):
                R[i, i] = 0.10

        # 额外约束（软约束，通过代价实现）：减少偶数电机（前/后竖直推进器）之间的极端不均衡。
        # 目的：避免出现“两个电机拉满、另外两个接近 0”这种不物理/易发散的分配。
        # 说明：这是软惩罚，不会禁止产生 pitch 力矩所需的前后差分，只是让优化器更偏好均衡解。
        def _add_diff_penalty(i, j, k):
            # 在 R 上加入 k*(u_i - u_j)^2
            R[i, i] += k
            R[j, j] += k
            R[i, j] -= k
            R[j, i] -= k

        def _add_linear_combo_penalty(indices, coeffs, k):
            # 在 R 上加入 k*(sum c_i*u_i)^2 = k * u^T (a a^T) u
            a = np.zeros(self.nu)
            for idx, c in zip(indices, coeffs):
                a[int(idx)] = float(c)
            R[:, :] = R + k * np.outer(a, a)

        # 四个竖直推进器在控制向量中的索引（对应 motor0,motor2,motor4,motor6）
        idx_front_up = 0
        idx_rear_up = 2
        idx_front_down = 4
        idx_rear_down = 6

        # 1) 同轴一前一后两桨尽量同速（防止只用其中一个）
        k_coax = float(get_value(cfg, "nmpc.penalties.k_coax", 0.03))
        _add_diff_penalty(idx_front_up, idx_front_down, k_coax)
        _add_diff_penalty(idx_rear_up, idx_rear_down, k_coax)

        # 2) 前后总推力尽量均衡： (front_up + front_down) - (rear_up + rear_down) ≈ 0
        #    这会减少“前面两桨打满、后面两桨几乎不转”的情况，但仍允许产生必要的前后差分。
        k_front_rear = float(get_value(cfg, "nmpc.penalties.k_front_rear", 0.02))
        _add_linear_combo_penalty(
            [idx_front_up, idx_front_down, idx_rear_up, idx_rear_down],
            [1.0, 1.0, -1.0, -1.0],
            k_front_rear,
        )

        # 舵机权重：希望机体水平/roll≈0 时舵机尽量回到各自中心（物理角≈pi/2）。
        # 之前“均值强、差动弱”的形式会让 (alpha-beta) 差动非常便宜，导致即使 roll≈0 也可能长期一高一低。
        # 改为分别惩罚 (alpha-center)^2 与 (beta-center)^2：
        # - 平时会更愿意把两舵机都压回中心
        # - 需要 roll 力矩时仍可差动，但代价会随差动增大而增大（更符合你的预期）
        # 如果 R_diag 已提供，则舵机权重由配置决定；否则保持历史默认。
        if not (isinstance(r_diag, list) and len(r_diag) == self.nu):
            k_servo = 100.0
            R[8, 8] = k_servo
            R[9, 9] = k_servo
            R[8, 9] = 0.0
            R[9, 8] = 0.0

        self.ocp.cost.W = scipy.linalg.block_diag(Q, R)

        Vx = np.zeros((self.ny, self.nx))
        Vx[0,0] = 1.0
        Vx[1,1] = 1.0
        Vx[2,2] = 1.0
        Vx[3,3] = 1.0
        Vx[4,4] = 1.0
        Vx[5,5] = 1.0
        Vx[6,6] = 1.0
        Vx[7,7] = 1.0
        Vx[8,8] = 1.0
        Vx[9,9] = 1.0
        Vx[10,10] = 1.0
        Vx[11,11] = 1.0
        Vx[12,12] = 1.0
        self.ocp.cost.Vx = Vx

        Vu = np.zeros((self.ny, self.nu))
        Vu[13,0] = 1.0
        Vu[14,1] = 1.0
        Vu[15,2] = 1.0
        Vu[16,3] = 1.0
        Vu[17,4] = 1.0
        Vu[18,5] = 1.0
        Vu[19,6] = 1.0
        Vu[20,7] = 1.0
        Vu[21,8] = 1.0
        Vu[22,9] = 1.0
        self.ocp.cost.Vu = Vu

        self.ocp.cost.W_e = float(get_value(cfg, "nmpc.W_e_scale", 50.0)) * Q

        Vx_e = np.zeros((self.ny_e, self.nx))
        Vx_e[0,0] = 1.0
        Vx_e[1,1] = 1.0
        Vx_e[2,2] = 1.0
        Vx_e[3,3] = 1.0
        Vx_e[4,4] = 1.0
        Vx_e[5,5] = 1.0
        Vx_e[6,6] = 1.0
        Vx_e[7,7] = 1.0
        Vx_e[8,8] = 1.0
        Vx_e[9,9] = 1.0
        Vx_e[10,10] = 1.0
        Vx_e[11,11] = 1.0
        Vx_e[12,12] = 1.0
        self.ocp.cost.Vx_e = Vx_e

        # 过程参考向量(状态+输入)
        # 重要：在存在浮力/水流等模型外稳态效应时，给电机一个固定的“悬停转速参考”会形成代价偏置，
        # 使优化器倾向保持某个非零推力而非精确把位置误差压回去，表现为缓慢漂移/高度爬升。
        # 因此这里对电机参考统一置 0，仅把舵机参考置于几何中心（更符合“roll≈0 时舵机回中”的设计意图）。
        self.ocp.cost.yref   = np.array([
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.servo_center_alpha, self.servo_center_beta
        ])
        # 终端参考向量(状态)
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 构建约束
        # 物理约束：电机转速非负；舵机以 pi/2 为中心小范围偏转（偏转差动用来平衡 roll）
        self.ocp.constraints.lbu = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            self.servo_center_alpha - self.servo_max_dev,
            self.servo_center_beta - self.servo_max_dev,
        ])
        self.ocp.constraints.ubu = np.array([
            self.max_speed, self.max_speed, self.max_speed, self.max_speed,
            self.max_speed, self.max_speed, self.max_speed, self.max_speed,
            self.servo_center_alpha + self.servo_max_dev,
            self.servo_center_beta + self.servo_max_dev,
        ])
        self.ocp.constraints.x0  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始状态
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 所有电机转速参与评估

        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0

        # set prediction horizon
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 显然更快 ~100Hz
        # self.ocp.solver_options.nlp_solver_type = 'SQP'  # ~10Hz

        self.ocp.model = self.model  # 传入模型
        
        # 设置参数初始值 (扰动力和力矩: [fx, fy, fz, mx, my, mz])
        self.ocp.parameter_values = np.zeros(6)  # 默认零扰动

        # 记录上一帧期望航向，避免目标在正上方/脚下时 yaw 角抖动
        self.last_desired_yaw = 0.0

        # 记录上一帧期望四元数，用于符号连续化（q 与 -q 表示同一姿态）
        self.last_desired_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # 调试开关：True 时打印参考信息（高频打印会明显拖慢仿真）
        self.debug = False
        
        # 构建编译OCP求解器
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file = 'acados_ocp.json')
        print("NMPC Controller Init Done")

    def referen_state_transition(self, current_state, goal_state):
        """分两步靠னர்目标: 先稳定偏航，再平移到目标。

        目的：避免 goal_state 姿态恒为 yaw=0 导致偏航在换点/机动时发散，
        并在目标几乎正上方/脚下时冻结期望 yaw 防止抖动。
        """

        def _yaw_from_quat(q):
            qw, qx, qy, qz = q
            return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        def _quat_from_yaw(yaw):
            half = 0.5 * yaw
            return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])

        def _wrap_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        def _unwrap_angle(prev, cur_wrapped):
            return prev + _wrap_pi(cur_wrapped - prev)

        pos_cur = current_state[0:3]
        quat_cur = current_state[3:7]
        pos_goal = goal_state[0:3]

        target_vec = pos_goal - pos_cur
        planar_norm = np.linalg.norm(target_vec[:2])

        current_yaw = _yaw_from_quat(quat_cur)

        # xy 太近：冻结 yaw（沿用上一帧期望），防止抖动
        yaw_deadband = 0.10  # m
        v_ref = goal_state[7:10]
        v_norm = float(np.linalg.norm(v_ref[:2]))
        if planar_norm < yaw_deadband:
            if v_norm > 1e-3:
                desired_yaw = self.last_desired_yaw
                if self.debug:
                    print("  >> XY近距且有参考速度，冻结期望 yaw")
            else:
                desired_yaw_quat = _yaw_from_quat(goal_state[3:7])
                desired_yaw = _unwrap_angle(self.last_desired_yaw, desired_yaw_quat)
                if self.debug:
                    print("  >> XY近距且无参考速度，使用目标向量期望 yaw: ", desired_yaw)
            self.last_desired_yaw = desired_yaw
        else:
            # 优先用参考速度方向作为期望 yaw（圆轨迹更自然，且避免“追点”带来的相位滞后）
            if v_norm > 1e-3:
                desired_yaw_wrapped = np.arctan2(v_ref[1], v_ref[0])
                if self.debug:
                    print("  >> 使用参考速度方向作为期望 yaw")
            else:
                desired_yaw_wrapped = np.arctan2(target_vec[1], target_vec[0])
                if self.debug:
                    print("  >> 使用目标向量方向作为期望 yaw")

            desired_yaw = _unwrap_angle(self.last_desired_yaw, desired_yaw_wrapped)
            self.last_desired_yaw = desired_yaw

        yaw_err = _wrap_pi(desired_yaw - current_yaw)
        yaw_threshold = np.deg2rad(10.0)
        yaw_freeze_max = np.deg2rad(45.0)

        new_goal = goal_state.copy()
        desired_quat = _quat_from_yaw(desired_yaw)

        # 四元数符号连续化：避免从 [0,0,0,1] 突然跳到 [0,0,0,-1] 这类“等价但数值不连续”的翻转
        if float(np.dot(desired_quat, self.last_desired_quat)) < 0.0:
            desired_quat = -desired_quat
        self.last_desired_quat = desired_quat

        new_goal[3:7] = desired_quat

        # 动态参考（例如圆轨迹）时不要用 yaw_err 去“冻结/缩小”平移目标。
        # 否则会出现：某些相位 yaw 误差稍大 → alpha 很小 → 平移几乎停滞；
        # 等 yaw 跟上后再突然追赶，视觉上像“半圈停住再继续”。
        v_ref_xy_norm = float(np.linalg.norm(goal_state[7:9]))
        is_dynamic_ref = v_ref_xy_norm > 1e-3

        # 偏航误差大时降低平移意图（仅用于静态“追点”目标）
        yaw_abs = abs(yaw_err)
        if (not is_dynamic_ref) and (yaw_abs > yaw_threshold):
            if yaw_abs >= yaw_freeze_max:
                alpha = 0.0
            else:
                alpha = 1.0 - (yaw_abs - yaw_threshold) / (yaw_freeze_max - yaw_threshold)
            new_goal[0:3] = pos_cur + alpha * (pos_goal - pos_cur)
            new_goal[7:10] = alpha * new_goal[7:10]
        new_goal[10:13] = 0.0

        if self.debug:
            print(f"New Goal Position: {new_goal[0:3]}", f"  Current Position: {pos_cur}")
        return new_goal

    # 状态空间位点控制
    # current_state当前状态: [x, y, z, qw, qx, qy, qz, vbx, vby, vbz, wx, wy, wz] 
    # goal_state目标状态:    [x, y, z, qw, qx, qy, qz, vbx, vby, vbz, wx, wy, wz]
    # disturbance扰动估计: [fx, fy, fz, mx, my, mz] (可选)
    def nmpc_state_control(self, current_state, goal_state, disturbance=None):
        _start = time.perf_counter()
        # Set initial condition, equality constraint
        self.acados_solver.set(0, 'lbx', current_state)
        self.acados_solver.set(0, 'ubx', current_state)

        goal_state = self.referen_state_transition(current_state, goal_state)

        y_ref = np.concatenate((goal_state, np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.servo_center_alpha, self.servo_center_beta
        ])))
        # Set Goal State
        for i in range(self.N):
            self.acados_solver.set(i, 'yref', y_ref)   # 过程参考
        y_refN = goal_state 
        self.acados_solver.set(self.N, 'yref', y_refN)   # 终端参考

        # 设置扰动参数 (如果提供)
        if disturbance is not None:
            # disturbance = [fx, fy, fz, mx, my, mz]
            d = np.asarray(disturbance, dtype=float).reshape(6,)
            d[0:3] = np.clip(d[0:3], -self.dist_force_clip, self.dist_force_clip)
            d[3:6] = np.clip(d[3:6], -self.dist_torque_clip, self.dist_torque_clip)
            self._disturbance_filt = (1.0 - self.dist_lpf_alpha) * self._disturbance_filt + self.dist_lpf_alpha * d
            p = self._disturbance_filt
        else:
            # 默认零扰动（并清空滤波状态，避免上一次 ESO 影响残留）
            self._disturbance_filt[:] = 0.0
            p = self._disturbance_filt
        
        # 为所有预测步长设置扰动参数
        for i in range(self.N):
            self.acados_solver.set(i, 'p', p)

        # Solve Problem
        status = self.acados_solver.solve()
        
        # 检查求解器状态
        if status != 0:
            print(f"⚠️ ACADOS求解失败! status={status}")
            if status == 1:
                print("   → 求解器达到最大迭代次数")
            elif status == 2:
                print("   → 求解器初始化失败")
            elif status == 3:
                print("   → QP求解器失败")
            elif status == 4:
                print("   → 求解器达到最大迭代次数（未收敛）")
            # 回退控制：
            # - 舵机保持几何中心（物理角约 pi/2）
            # - 仅 1/3/5/7 四个竖直推进器承担重力
            # - 为避免“摔倒后卡在地面、只给悬停推力起不来”，对 z 做一个小的 PD，给出略大于悬停的推力
            _end = time.perf_counter()
            _dt = _end - _start

            z = float(current_state[2])
            vz = float(current_state[9])
            z_ref = float(goal_state[2])
            vz_ref = float(goal_state[9])

            # 经验值：只在求解失败时启用，保守一点即可
            kp_z = 2.5
            kd_z = 1.2
            a_z_cmd = kp_z * (z_ref - z) + kd_z * (vz_ref - vz)
            a_z_cmd = float(np.clip(a_z_cmd, -2.0, 6.0))
            thrust_total = self.mq * (self.g0 + a_z_cmd)
            thrust_per = float(np.clip(thrust_total / 4.0, 0.0, 4.0 * 24.0))
            w_fb = float(np.sqrt(max(thrust_per, 0.0) / self.Ct))
            w_fb = float(np.clip(w_fb, 0.0, self.max_speed))

            return _dt, np.array([
                w_fb, 0.0, w_fb, 0.0, w_fb, 0.0, w_fb, 0.0,
                self.servo_center_alpha, self.servo_center_beta
            ])
        
        # Get Solution (仅在成功时获取)
        w_opt_acados = np.ndarray((self.N, self.nu))  # 控制输入
        x_opt_acados = np.ndarray((self.N + 1, len(current_state)))   # 状态估计
        x_opt_acados[0, :] = self.acados_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_solver.get(i + 1, "x")
        # return w_opt_acados, x_opt_acados  # 返回控制输入和状态
        _end = time.perf_counter()
        _dt = _end - _start
        return _dt, w_opt_acados[0]  # 返回最近控制输入 10 维向量
        # control_input = self.acados_solver.get(0, "u")
        # state_estimate = self.acados_solver.get(self.N, "x")
        # return control_input, state_estimate  # 返回所有控制输入和状态

    # NMPC位置控制
    # goal_pos: 目标三维位置[x y z]
    def nmpc_position_control(self, current_state, goal_pos, disturbance=None, goal_vel=None):
        """
        位置控制
        
        Args:
            current_state: 当前状态 [x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz]
            goal_pos: 目标位置 [x,y,z]
            disturbance: 扰动估计 dict {'force': [fx,fy,fz], 'torque': [mx,my,mz]}
                        可选，直接用于MPC模型中
        
        Returns:
            _dt: 求解时间
            control: 控制输出 [w1..w8, alpha, beta]
        """
        if goal_vel is None:
            goal_vel = np.zeros(3)
        goal_vel = np.asarray(goal_vel, dtype=float).reshape(3,)

        # goal_state: [x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz]
        # 重要：对圆轨迹/方轨迹这类“动态参考”，给 vx/vy/vz 前馈能避免 referen_state_transition()
        # 把它误判成“静态追点”而在某些相位抑制平移（表现为只跑一小段就停）。
        goal_state = np.array([
            goal_pos[0], goal_pos[1], goal_pos[2],
            1.0, 0.0, 0.0, 0.0,
            goal_vel[0], goal_vel[1], goal_vel[2],
            0.0, 0.0, 0.0,
        ])
        
        # 将扰动字典转换为参数数组 [fx, fy, fz, mx, my, mz]
        if disturbance is not None:
            dist_array = np.concatenate([disturbance['force'], disturbance['torque']])
        else:
            dist_array = None
        
        # 将扰动参数传递给MPC求解器
        _dt, control = self.nmpc_state_control(current_state, goal_state, dist_array)
        
        # 记录数据: [time, state(13), goal(3), control(10), solve_time, disturbance]
        import time
        log_entry = {
            'time': time.time(),
            'state': current_state.copy(),
            'goal': goal_pos.copy(),
            'control': control.copy(),
            'solve_time': _dt
        }
        
        # 记录扰动估计（如果有）
        if disturbance is not None:
            log_entry['dist_force'] = disturbance['force'].copy()
            log_entry['dist_torque'] = disturbance['torque'].copy()
        else:
            log_entry['dist_force'] = np.zeros(3)
            log_entry['dist_torque'] = np.zeros(3)
        
        self.data_log.append(log_entry)
        
        return _dt, control
    
    def save_data(self, filename='nmpc_data.csv'):
        """保存数据到CSV文件"""
        import csv
        if not self.data_log:
            print("没有数据可保存")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头（包含扰动估计）
            header = ['time',
                     'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz',
                     'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                     'goal_x', 'goal_y', 'goal_z',
                     'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u_alpha', 'u_beta',
                     'solve_time',
                     'dist_fx', 'dist_fy', 'dist_fz',
                     'dist_mx', 'dist_my', 'dist_mz']
            writer.writerow(header)
            
            # 写入数据
            t0 = self.data_log[0]['time']
            for data in self.data_log:
                row = [data['time'] - t0]  # 相对时间
                row.extend(data['state'])
                row.extend(data['goal'])
                row.extend(data['control'])
                row.append(data['solve_time'])
                # 添加扰动估计
                row.extend(data['dist_force'])
                row.extend(data['dist_torque'])
                writer.writerow(row)
        
        print(f"✓ 数据已保存到: {filename} ({len(self.data_log)} 条记录)")
    
    def clear_data(self):
        """清空数据日志"""
        self.data_log = []

# TEST
if __name__ == '__main__':
    print("20250304 ACADOS NMPC TEST")
    nmpc_controller = NMPC_Controller()
    current_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_state = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _dt, w = nmpc_controller.nmpc_state_control(current_state, goal_state)

    # print("Control Input:")
    # print(w)
    # print("State Estimation:")
    # print(x)
