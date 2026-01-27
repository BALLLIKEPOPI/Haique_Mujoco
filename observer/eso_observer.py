import numpy as np
import csv

from typing import Optional
from os.path import abspath, dirname, join

from model.config_loader import get_mode_config, get_value

class ESO_Observer:
    def __init__(
        self,
        dt: float = 0.01,
        mode: str = "aerial",
        left_servo_offset: Optional[float] = None,
        right_servo_offset: Optional[float] = None,
        config_path: str = join(dirname(abspath(__file__)), "config.yaml"),
    ):
        # 1. 物理参数（从配置读取，保证与模型/NMPC 一致）
        self.mode = str(mode)
        self.dt = float(dt)

        cfg = get_mode_config(self.mode, path=config_path)

        self.g0 = float(get_value(cfg, "physical.g", 9.8066))
        
        # 水下浮力 (仅underwater模式)
        rho_water = 1000.0
        displaced_volume = 0.004774  # m^3
        self.buoyancy = rho_water * self.g0 * displaced_volume  # 浮力 (N)
        self.mass = float(get_value(cfg, "physical.mass", 4.672))
        inertia = get_value(cfg, "physical.inertia", [0.10170715, 0.10222875, 0.16095642])
        self.inertia = np.asarray(inertia, dtype=float)
        self.Ct = float(get_value(cfg, "physical.Ct", 0.0267))
        self.Cd = float(get_value(cfg, "physical.Cd", 0.00111))
        self.dq = float(get_value(cfg, "physical.dq", 0.605))
        self.l = self.dq / 2.0

        # MuJoCo 中实际舵机角 = 控制输出 + offset。
        # 这里保存 offset，供 underwater 模式下计算“物理角”。
        cfg_left_offset = float(get_value(cfg, "servo.left_offset", -0.1))
        cfg_right_offset = float(get_value(cfg, "servo.right_offset", 0.0))
        self.left_servo_offset = float(cfg_left_offset if left_servo_offset is None else left_servo_offset)
        self.right_servo_offset = float(cfg_right_offset if right_servo_offset is None else right_servo_offset)

        # yaw mixing factors
        self.k_yaw_lr = float(get_value(cfg, "model.k_yaw_lr", 1.5))
        self.k_yaw_other = float(get_value(cfg, "model.k_yaw_other", 0.3))

        # export_model.py uses yaw bias mixing; keep default unless configured later.
        self.k_yaw = 0.8

        # 2. 状态变量 [z1: 状态跟踪, z2: 扰动估计]
        # 0-2: vx, vy, vz (世界坐标系线速度)
        # 3-5: wx, wy, wz (机体坐标系角速度)
        self.z1 = np.zeros(6)
        self.z2 = np.zeros(6)

        # 3. NLESO 增益参数 (需要根据实际表现微调)
        omega_pos = float(get_value(cfg, "eso.omega_pos", 15.0))  # 位置/速度观测带宽
        omega_att = float(get_value(cfg, "eso.omega_att", 25.0))  # 姿态/角速度观测带宽
        self.beta1 = np.array([2*omega_pos]*3 + [2*omega_att]*3)
        self.beta2 = np.array([omega_pos**2]*3 + [omega_att**2]*3)

        self.data_log = []

    def _fal(self, e, alpha, zeta):
        """非线性函数 fal，增强小误差时的灵敏度"""
        if abs(e) > zeta:
            return np.sign(e) * (abs(e)**alpha)
        else:
            return e / (zeta**(1 - alpha))

    def _get_control_effect(self, u, quat):
        if self.mode == 'aerial':
            return self._get_control_effect_aerial(u, quat)
        elif self.mode == 'underwater':
            return self._get_control_effect_underwater(u, quat)

    def _get_control_effect_aerial(self, u, quat):
        """
        根据输入 u=[w1, w2, w3, w4, yaw_bias] 和当前姿态计算理论加速度和力矩
        逻辑严格对应 export_model.py
        """
        if u is None or len(u) < 5:
            return np.zeros(3), np.zeros(3)
        
        w1, w2, w3, w4, yaw_bias = u[:5]
        
        # 混控逻辑：计算8个电机的转速
        w1_ = w1 * (1 - self.k_yaw * yaw_bias)
        w2_ = w2 * (1 + self.k_yaw * yaw_bias)
        w3_ = w3 * (1 - self.k_yaw * yaw_bias)
        w4_ = w4 * (1 + self.k_yaw * yaw_bias)
        w5_ = w1 * (1 + self.k_yaw * yaw_bias)
        w6_ = w2 * (1 - self.k_yaw * yaw_bias)
        w7_ = w3 * (1 + self.k_yaw * yaw_bias)
        w8_ = w4 * (1 - self.k_yaw * yaw_bias)

        # 计算理论推力加速度 (机体系 Z 轴)
        thrust_acc_b = self.Ct * (w1_**2 + w2_**2 + w3_**2 + w4_**2 + 
                                  w5_**2 + w6_**2 + w7_**2 + w8_**2) / self.mass
        
        # 将机体系推力加速度转换至世界系
        q0, q1, q2, q3 = quat
        acc_w = np.array([
            2 * (q1*q3 + q0*q2) * thrust_acc_b,
            2 * (-q0*q1 + q2*q3) * thrust_acc_b,
            2 * (0.5 - q1**2 - q2**2) * thrust_acc_b - self.g0
        ])

        # 计算理论机体系力矩
        mx = self.Ct * self.l * (w2_**2 + w6_**2 - w4_**2 - w8_**2)
        my = self.Ct * self.l * (-w1_**2 - w5_**2 + w3_**2 + w7_**2)
        mz = self.Cd * (-w1_**2 + w2_**2 - w3_**2 + w4_**2 + 
                        w5_**2 - w6_**2 + w7_**2 - w8_**2)
        
        # 转换为角加速度以匹配b0定义
        torque_b = np.array([mx, my, mz])
        angular_acc_b = torque_b / np.array(self.inertia)
        
        return acc_w, angular_acc_b

    def _get_control_effect_underwater(self, u, quat):
        """Underwater: compute theoretical acceleration (world) and torques (body).

        u = [w1..w8, alpha_cmd, beta_cmd]
        NOTE: MuJoCo 实际舵机角 = cmd + offset，因此这里必须用物理角参与三角分解。
        """
        if u is None or len(u) < 10:
            return np.zeros(3), np.zeros(3)

        w = np.asarray(u[:8], dtype=float)
        alpha_cmd = float(u[8])
        beta_cmd = float(u[9])

        alpha = alpha_cmd + self.left_servo_offset
        beta = beta_cmd + self.right_servo_offset

        # thrust per motor
        f = self.Ct * w * np.abs(w)

        # reaction torques per motor (signs follow export_model_underwater.py)
        m = np.zeros(8)
        m[0] = -self.Cd * (w[0] ** 2)
        m[1] = self.Cd * w[1] * np.abs(w[1])
        m[2] = -self.Cd * (w[2] ** 2)
        m[3] = self.Cd * w[3] * np.abs(w[3])
        m[4] = self.Cd * (w[4] ** 2)
        m[5] = -self.Cd * w[5] * np.abs(w[5])
        m[6] = self.Cd * (w[6] ** 2)
        m[7] = -self.Cd * w[7] * np.abs(w[7])

        # body forces
        fx_b = f[1] * np.sin(alpha) + f[3] * np.sin(beta) + f[5] * np.sin(alpha) + f[7] * np.sin(beta)
        fy_b = 0.0
        fz_b = (
            f[0] + f[1] * np.cos(alpha) + f[2] + f[3] * np.cos(beta)
            + f[4] + f[5] * np.cos(alpha) + f[6] + f[7] * np.cos(beta)
        )

        # rotate body force to world and divide by mass
        q0, q1, q2, q3 = quat
        r00 = 1 - 2 * (q2**2 + q3**2)
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 1 - 2 * (q1**2 + q3**2)
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 1 - 2 * (q1**2 + q2**2)

        acc_w = np.array([
            (r00 * fx_b + r01 * fy_b + r02 * fz_b) / self.mass,
            (r10 * fx_b + r11 * fy_b + r12 * fz_b) / self.mass,
            (r20 * fx_b + r21 * fy_b + r22 * fz_b) / self.mass - self.g0 + self.buoyancy / self.mass,
        ])

        # body torques
        mx = self.l * self.Ct * ((w[1] ** 2 + w[5] ** 2) * np.cos(alpha) - (w[3] ** 2 + w[7] ** 2) * np.cos(beta)) + (
            m[1] * np.sin(alpha) + m[5] * np.sin(alpha) + m[3] * np.sin(beta) + m[7] * np.sin(beta)
        )
        my = self.l * self.Ct * (-w[0] ** 2 - w[4] ** 2 + w[2] ** 2 + w[6] ** 2)

        mz_lr = (
            -self.l * ((f[1] + f[5]) * np.sin(alpha) - (f[3] + f[7]) * np.sin(beta))
            + (m[1] + m[5]) * np.cos(alpha)
            + (m[3] + m[7]) * np.cos(beta)
        )
        mz_other = (m[0] + m[2] + m[4] + m[6])
        mz = self.k_yaw_lr * mz_lr + self.k_yaw_other * mz_other

        # 转换为角加速度以匹配b0定义
        torque_b = np.array([mx, my, mz])
        angular_acc_b = torque_b / np.array(self.inertia)

        return acc_w, angular_acc_b

    def update(self, state_obs, last_u, quat):
        """
        更新观测器
        state_obs: [vx, vy, vz, wx, wy, wz] (MuJoCo 传感器数据)
        last_u: 上一时刻控制量 
               Aerial: [w1, w2, w3, w4, yaw_bias]
               Underwater: [w1..w8, alpha, beta]
        quat: 当前姿态四元数 [q0, q1, q2, q3]
        """
        acc_theory_w, angular_acc_theory_b = self._get_control_effect(last_u, quat)
        
        # 陀螺力矩补偿 (gyroscopic effect compensation)
        wx, wy, wz = state_obs[3:6]
        gyro_acc = np.array([
            (self.inertia[1] - self.inertia[2]) * wy * wz / self.inertia[0],  # 陀螺角加速度 x
            (self.inertia[2] - self.inertia[0]) * wx * wz / self.inertia[1],  # 陀螺角加速度 y
            (self.inertia[0] - self.inertia[1]) * wx * wy / self.inertia[2]   # 陀螺角加速度 z
        ])
        
        # 将陀螺角加速度加入控制输入，让ESO只观测外部扰动
        angular_acc_with_gyro = angular_acc_theory_b + gyro_acc
        
        # 对应 export_model.py 的输入增益 b0
        b0 = np.array([
            1.0, 1.0, 1.0,              # 线速度通道 (m/s²)
            1.0, 1.0, 1.0               # 角速度通道 (rad/s²) - 已在_get_control_effect中转换
        ])
        
        # 理论输入向量 U
        # 注意：这里对于角速度通道，由于 export_model 中有陀螺力矩项，ESO 观测的是剩余扰动
        u_vec = np.concatenate([acc_theory_w, angular_acc_with_gyro])

        for i in range(6):
            e = self.z1[i] - state_obs[i]
            fe = self._fal(e, 0.5, self.dt)
            fe1 = self._fal(e, 0.25, self.dt)

            # 更新 z1 (状态跟踪) 和 z2 (扩张扰动状态)
            self.z1[i] += self.dt * (self.z2[i] - self.beta1[i] * fe + b0[i] * u_vec[i])
            self.z2[i] += self.dt * (-self.beta2[i] * fe1)

            # 饱和限幅：防止初始暂态或模型误差导致的z2跳变
            # 力通道：限制加速度扰动在 ±5 m/s^2 (对应 ±35 N)
            # 力矩通道：限制角加速度扰动在 ±20 rad/s^2 (对应 ±2 Nm)
            if i < 3:
                self.z2[i] = np.clip(self.z2[i], -2.0, 2.0)  # 力通道
            else:
                self.z2[i] = np.clip(self.z2[i], -10.0, 10.0)  # 力矩通道

        dist_f = self.z2[:3] * self.mass  
        dist_m = self.z2[3:6] * self.inertia
        self.data_log.append(np.concatenate([dist_f, dist_m]))

        # 返回 dist_f (世界系) 和 dist_m (机体系)
        # 注意：dist_f 在 export_model 中是力，所以需要乘上质量
        return dist_f, dist_m
    
    def save_disturbance_log(self, filename='eso_disturbance_log.csv'):
        if not self.data_log:
            print("ESO日志为空，未保存数据")
            return

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['dist_fx', 'dist_fy', 'dist_fz', 'dist_mx', 'dist_my', 'dist_mz'])
                # 写入所有行
                writer.writerows(self.data_log)
            print(f"✓ ESO扰动数据已保存到 {filename}")
        except Exception as e:
            print(f"保存ESO数据失败: {e}")
        