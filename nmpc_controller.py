# 四旋翼NMPC控制器 20250304 Wakkk
# ACADOS NMPC
from acados_template import AcadosOcp, AcadosOcpSolver
from export_model import *
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
    def __init__(self, config_path: str = join(dirname(abspath(__file__)), "config.yaml")):
        self.ocp = AcadosOcp()       # OCP 优化问题

        cfg = get_mode_config("aerial", path=config_path)

        # physical/model params
        self.g0 = float(get_value(cfg, "physical.g", 9.8066))
        self.mq = float(get_value(cfg, "physical.mass", 4.672))
        inertia = get_value(cfg, "physical.inertia", [0.10170715, 0.10222875, 0.16095642])
        self.inertia = np.asarray(inertia, dtype=float)
        self.Ct = float(get_value(cfg, "physical.Ct", 0.1757))
        self.Cd = float(get_value(cfg, "physical.Cd", 0.02))
        self.dq = float(get_value(cfg, "physical.dq", 0.605))
        self.k_yaw = float(get_value(cfg, "model.k_yaw", 0.8))

        # self.model = export_model()  # 导出四旋翼物理模型
        self.model = export_model(
            g0=self.g0,
            mass=self.mq,
            inertia=tuple(float(x) for x in self.inertia.tolist()),
            Ct=self.Ct,
            Cd=self.Cd,
            dq=self.dq,
            k_yaw=self.k_yaw,
        )
        
        # 数据记录
        self.data_log = []

        self.Tf = float(get_value(cfg, "nmpc.Tf", 0.6))          # 预测时间长度(s)
        self.N = int(get_value(cfg, "nmpc.N", 30))              # 预测步数(节点数量)
        self.nx = self.model.x.size()[0]    # 状态维度 13维度
        self.nu = self.model.u.size()[0]    # 控制输入维度 5维度 (w1,w2,w3,w4,yaw_bias)
        self.ny = self.nx + self.nu         # 评估维度
        self.ny_e = self.nx                 # 终端评估维度

        # set ocp_nlp_dimensions
        self.nlp_dims     = self.ocp.dims
        self.nlp_dims.N   = self.N

        # bounds
        # 注意：本模型里 NMPC 的 4 个“基准电机转速”会通过 yaw_bias 混控生成 8 个实际电机转速参与推力。
        # yaw_bias=0 且 w1=w2=w3=w4=w 时，总推力 = Ct * 8 * w^2，因此悬停应使用 8 电机公式。
        self.hov_w = np.sqrt((self.mq * self.g0) / (8 * self.Ct))
        print(f"hovor speed: {self.hov_w} krpm")
        # hovor speed: 15.777730167256925 krpm

        # MuJoCo 执行端 calc_motor_input() 会将电机转速硬限幅到 22krpm。
        # 约束需要与执行端一致，否则 NMPC 会规划不可实现的推力，导致 z 通道振荡/发散。
        self.max_speed = float(get_value(cfg, "nmpc.max_speed", 22.0))

        # set weighting matrices 状态权重矩阵
        q_diag = get_value(cfg, "nmpc.Q_diag", None)
        if isinstance(q_diag, list) and len(q_diag) == self.nx:
            Q = np.diag(np.asarray(q_diag, dtype=float))
        else:
            Q = np.eye(self.nx)
            Q[0,0] = 180.0       # x
            Q[1,1] = 180.0       # y
            Q[2,2] = 260.0       # z
            Q[3,3] = 10.0        # qw
            Q[4,4] = 60.0        # qx
            Q[5,5] = 80.0        # qy
            Q[6,6] = 80.0        # qz
            Q[7,7] = 80.0        # vx
            Q[8,8] = 80.0        # vy
            Q[9,9] = 260.0       # vz
            Q[10,10] = 120.0     # wx
            Q[11,11] = 120.0     # wy
            Q[12,12] = 80.0      # wz

        r_diag = get_value(cfg, "nmpc.R_diag", None)
        if isinstance(r_diag, list) and len(r_diag) == self.nu:
            R = np.diag(np.asarray(r_diag, dtype=float))
        else:
            R = np.eye(self.nu)   # 控制输入权重矩阵
            R[0,0] = 0.95
            R[1,1] = 0.95
            R[2,2] = 0.95
            R[3,3] = 0.95
            R[4,4] = 0.40

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
        self.ocp.cost.Vu = Vu

        self.ocp.cost.W_e = float(get_value(cfg, "nmpc.W_e_scale", 10.0)) * Q

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

        # 过程参考向量(状态+输入) —— 4 桨悬停 + yaw_bias=0
        self.ocp.cost.yref   = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.hov_w, self.hov_w, self.hov_w, self.hov_w, 0.0])
        # 终端参考向量(状态)
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 构建约束
        self.ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0, -0.8])  # yaw_bias限制在±0.5（配合k_yaw=0.3，允许±15%调整）
        self.ocp.constraints.ubu = np.array([+self.max_speed,+self.max_speed,+self.max_speed,+self.max_speed, 0.8])  # 避免过度不平衡
        self.ocp.constraints.x0  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始状态
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])  # 所有电机转速参与评估

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
            # v_ref = goal_state[7:10]
            # v_norm = float(np.linalg.norm(v_ref[:2]))
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
        # 记录“实际送进NMPC”的参考，用于日志/可视化
        self._last_goal_state_used = goal_state.copy()

        y_ref = np.concatenate((goal_state, np.array([
            self.hov_w, self.hov_w, self.hov_w, self.hov_w, 0.0
        ])))
        # Set Goal State
        for i in range(self.N):
            self.acados_solver.set(i, 'yref', y_ref)   # 过程参考
        y_refN = goal_state 
        self.acados_solver.set(self.N, 'yref', y_refN)   # 终端参考

        # 设置扰动参数 (如果提供)
        if disturbance is not None:
            # disturbance = [fx, fy, fz, mx, my, mz]
            p = disturbance
        else:
            # 默认零扰动
            p = np.zeros(6)
        
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
            # 返回悬停控制量作为安全回退
            _end = time.perf_counter()
            _dt = _end - _start
            return _dt, np.array([self.hov_w, self.hov_w, self.hov_w, self.hov_w, 0.0])
        
        # Get Solution (仅在成功时获取)
        w_opt_acados = np.ndarray((self.N, 5))  # 控制输入
        x_opt_acados = np.ndarray((self.N + 1, len(current_state)))   # 状态估计
        x_opt_acados[0, :] = self.acados_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_solver.get(i + 1, "x")
        # return w_opt_acados, x_opt_acados  # 返回控制输入和状态
        _end = time.perf_counter()
        _dt = _end - _start
        return _dt, w_opt_acados[0]  # 返回最近控制输入 4 Vector
        # control_input = self.acados_solver.get(0, "u")
        # state_estimate = self.acados_solver.get(self.N, "x")
        # return control_input, state_estimate  # 返回所有控制输入和状态

    # NMPC位置控制
    # goal_pos: 目标三维位置[x y z]
    def nmpc_position_control(self, current_state, goal_pos, goal_vel=None, disturbance=None):
        """
        位置控制
        
        Args:
            current_state: 当前状态 [x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz]
            goal_pos: 目标位置 [x,y,z]
            goal_vel: 目标速度 [vx,vy,vz]（可选，用于移动参考前馈）
            disturbance: 扰动估计 dict {'force': [fx,fy,fz], 'torque': [mx,my,mz]}（可选）
        
        Returns:
            _dt: 求解时间
            control: 控制输出 [w_front, w_left, w_rear, w_right, yaw_bias]
        """
        # 兼容旧签名：第三个参数若是 disturbance dict，则视为 disturbance
        if goal_vel is not None and isinstance(goal_vel, dict):
            disturbance = goal_vel
            goal_vel = None

        if goal_vel is None:
            goal_vel = np.zeros(3)

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

        # 实际使用的参考（经过 referen_state_transition 后）
        goal_used = getattr(self, '_last_goal_state_used', goal_state)
        
        # 记录数据: [time, state(13), goal(3), control(5), solve_time, disturbance]
        import time
        log_entry = {
            'time': time.time(),
            'state': current_state.copy(),
            'goal': goal_pos.copy(),
            'goal_used': goal_used[0:3].copy(),
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
                     'goal_used_x', 'goal_used_y', 'goal_used_z',
                     'u_front', 'u_left', 'u_rear', 'u_right', 'u_yaw_bias',
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
                row.extend(data.get('goal_used', data['goal']))
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
