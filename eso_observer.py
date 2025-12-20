import numpy as np
import csv

class ESO_Observer:
    def __init__(self, dt=0.01, mode='aerial'):
        # 1. 物理参数
        self.mode = mode
        self.g0 = 9.8066
        self.mass = 4.672
        self.dt = dt

        if self.mode == 'aerial':
            # 严格对应 export_model.py
            self.inertia = np.array([0.10170715, 0.10222875, 0.16095642])
            self.Ct = 0.1757
            self.Cd = 0.02
            self.l = 0.605 / 2.0
            self.k_yaw = 0.8
        elif self.mode == 'underwater':
            # 对应 export_model_underwater.py
            self.inertia = np.array([0.10170715, 0.10222875, 0.16095642])
            self.Ct = 0.1757
            self.Cd = 0.02
            self.dq = 0.605
            self.l = self.dq / 2.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. 状态变量 [z1: 状态跟踪, z2: 扰动估计]
        # 0-2: vx, vy, vz (世界坐标系线速度)
        # 3-5: wx, wy, wz (机体坐标系角速度)
        self.z1 = np.zeros(6)
        self.z2 = np.zeros(6)

        # 3. NLESO 增益参数 (需要根据实际表现微调)
        omega_pos = 15.0  # 位置/速度观测带宽
        omega_att = 25.0  # 姿态/角速度观测带宽
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
        
        return acc_w, np.array([mx, my, mz])

    def _get_control_effect_underwater(self, u, quat):
        """
        根据输入 u=[w1..w8, alpha, beta] 和当前姿态计算理论加速度和力矩
        逻辑严格对应 export_model_underwater.py
        """
        if u is None or len(u) < 10:
            return np.zeros(3), np.zeros(3)
        
        # u: w1, w2, w3, w4, w5, w6, w7, w8, alpha, beta
        w = u[:8]
        alpha = u[8]
        beta = u[9]

        # 计算 f1..f8 (Ct * w * |w|)
        f = self.Ct * w * np.abs(w)

        # 计算 m1..m8 (Cd/Ct * f or Cd*w*|w| depending on prop rotation)
        # export_model_underwater.py:
        # m1 = -Cd * w1**2 (approx w1*|w1|) -> sign matter? 
        # export_model: m1 = -Cd * w1^2. But w1 can be negative in underwater?
        # In export_model_underwater: m1 = -Cd * w1^2 if w1 is squared? 
        # Actually export_model_underwater.py says:
        # m1 = -Cd * w1**2
        # m2 = Cd * w2 * SX.fabs(w2)
        # ...
        # NOTE: w is usually positive for simple quad but underwater can reverse?
        # Let's strictly follow export_model_underwater logic.
        
        m = np.zeros(8)
        # Odd motors (1,3,5,7 in 1-based index -> 0,2,4,6 in 0-based)
        # Even motors (2,4,6,8 in 1-based index -> 1,3,5,7 in 0-based)
        
        # Index 0 (w1): m1 = -Cd * w1**2
        m[0] = -self.Cd * (w[0]**2)
        # Index 1 (w2): m2 = Cd * w2 * abs(w2)
        m[1] = self.Cd * w[1] * np.abs(w[1])
        # Index 2 (w3): m3 = -Cd * w3**2
        m[2] = -self.Cd * (w[2]**2)
        # Index 3 (w4): m4 = Cd * w4 * abs(w4)
        m[3] = self.Cd * w[3] * np.abs(w[3])
        # Index 4 (w5): m5 = Cd * w5**2  <-- Note: export_model says m5 = Cd * w5**2
        m[4] = self.Cd * (w[4]**2)
        # Index 5 (w6): m6 = -Cd * w6 * abs(w6)
        m[5] = -self.Cd * w[5] * np.abs(w[5])
        # Index 6 (w7): m7 = Cd * w7**2
        m[6] = self.Cd * (w[6]**2)
        # Index 7 (w8): m8 = -Cd * w8 * abs(w8)
        m[7] = -self.Cd * w[7] * np.abs(w[7])

        # Body forces
        # fx_b = f2*sin(a) + f4*sin(b) + f6*sin(a) + f8*sin(b)
        fx_b = f[1]*np.sin(alpha) + f[3]*np.sin(beta) + f[5]*np.sin(alpha) + f[7]*np.sin(beta)
        fy_b = 0.0
        # fz_b = f1 + f2*cos(a) + f3 + f4*cos(b) + f5 + f6*cos(a) + f7 + f8*cos(b)
        fz_b = f[0] + f[1]*np.cos(alpha) + f[2] + f[3]*np.cos(beta) + \
               f[4] + f[5]*np.cos(alpha) + f[6] + f[7]*np.cos(beta)

        # Rotate to World frame
        # quat: [q0, q1, q2, q3] -> [w, x, y, z]
        q0, q1, q2, q3 = quat
        
        # Rotation matrix Rwb
        # Row 0
        r00 = 1 - 2*(q2**2 + q3**2)
        r01 = 2*(q1*q2 - q0*q3)
        r02 = 2*(q1*q3 + q0*q2)
        # Row 1
        r10 = 2*(q1*q2 + q0*q3)
        r11 = 1 - 2*(q1**2 + q3**2)
        r12 = 2*(q2*q3 - q0*q1)
        # Row 2
        r20 = 2*(q1*q3 - q0*q2)
        r21 = 2*(q2*q3 + q0*q1)
        r22 = 1 - 2*(q1**2 + q2**2)

        acc_x = (r00*fx_b + r01*fy_b + r02*fz_b) / self.mass
        acc_y = (r10*fx_b + r11*fy_b + r12*fz_b) / self.mass
        acc_z = (r20*fx_b + r21*fy_b + r22*fz_b) / self.mass - self.g0

        acc_w = np.array([acc_x, acc_y, acc_z])

        # Body Torques
        # mx
        mx = self.l * self.Ct * ((w[1]**2 + w[5]**2)*np.cos(alpha) - (w[3]**2 + w[7]**2)*np.cos(beta)) + \
             m[1]*np.sin(alpha) + m[5]*np.sin(alpha) + m[3]*np.sin(beta) + m[7]*np.sin(beta)
        
        # my
        my = self.l * self.Ct * (-w[0]**2 - w[4]**2 + w[2]**2 + w[6]**2)

        # mz
        mz = -self.l * self.Ct * ((w[1]**2 + w[5]**2)*np.sin(alpha) - (w[3]**2 + w[7]**2)*np.sin(beta)) + \
             m[0] + m[2] + m[4] + m[6] + \
             m[1]*np.cos(alpha) + m[5]*np.cos(alpha) + \
             m[3]*np.cos(beta) + m[7]*np.cos(beta)

        return acc_w, np.array([mx, my, mz])

    def update(self, state_obs, last_u, quat):
        """
        更新观测器
        state_obs: [vx, vy, vz, wx, wy, wz] (MuJoCo 传感器数据)
        last_u: 上一时刻控制量 
               Aerial: [w1, w2, w3, w4, yaw_bias]
               Underwater: [w1..w8, alpha, beta]
        quat: 当前姿态四元数 [q0, q1, q2, q3]
        """
        acc_theory_w, torque_theory_b = self._get_control_effect(last_u, quat)
        
        # 对应 export_model.py 的输入增益 b0
        b0 = np.array([
            1.0, 1.0, 1.0,              # 线速度通道 (acc已处理过mass)
            1/self.inertia[0], 1/self.inertia[1], 1/self.inertia[2] # 角速度通道
        ])
        
        # 理论输入向量 U
        # 注意：这里对于角速度通道，由于 export_model 中有陀螺力矩项，ESO 观测的是剩余扰动
        u_vec = np.concatenate([acc_theory_w, torque_theory_b])

        for i in range(6):
            e = self.z1[i] - state_obs[i]
            fe = self._fal(e, 0.5, self.dt)
            fe1 = self._fal(e, 0.25, self.dt)

            # 更新 z1 (状态跟踪) 和 z2 (扩张扰动状态)
            self.z1[i] += self.dt * (self.z2[i] - self.beta1[i] * fe + b0[i] * u_vec[i])
            self.z2[i] += self.dt * (-self.beta2[i] * fe1)

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
        