#!/usr/bin/env python3
"""
正向动力学模型 - 基于 export_model_underwater.py
用于数值仿真，根据控制量计算状态导数
"""
import numpy as np


class ForwardDynamics:
    """基于标称模型的正向动力学"""
    
    def __init__(
        self,
        g0: float = 9.8066,
        mass: float = 4.672,
        inertia=(0.10170715, 0.10222875, 0.16095642),
        Ct: float = 0.0267,
        Cd: float = 0.00111,
        dq: float = 0.605,
        alpha_offset: float = -0.1,
        beta_offset: float = 0.0,
        k_yaw_lr: float = 1.5,
        k_yaw_other: float = 0.3,
        rho_water: float = 1000.0,
        displaced_volume: float = 0.004774,  # m^3，排水体积
    ):
        self.g0 = g0
        self.mass = mass
        self.Ixx, self.Iyy, self.Izz = inertia
        self.Ct = Ct
        self.Cd = Cd
        self.l = dq / 2.0
        self.alpha_offset = alpha_offset
        self.beta_offset = beta_offset
        self.k_yaw_lr = k_yaw_lr
        self.k_yaw_other = k_yaw_other
        
        # 水下浮力
        self.rho_water = rho_water
        self.V = displaced_volume
        self.buoyancy = rho_water * g0 * displaced_volume  # 浮力 (N)
        
    
    def compute_dynamics(self, state, control, disturbance=None):
        """
        计算状态导数
        
        Args:
            state: [px, py, pz, q0, q1, q2, q3, vx, vy, vz, wx, wy, wz] (13维)
            control: [w1~w8, alpha, beta] (10维)
            disturbance: {'force': [fx,fy,fz], 'torque': [mx,my,mz]} (可选)
        
        Returns:
            state_dot: 状态导数 (13维)
        """
        # 解包状态
        px, py, pz = state[0:3]
        q0, q1, q2, q3 = state[3:7]
        vx, vy, vz = state[7:10]
        wx, wy, wz = state[10:13]
        
        # 解包控制
        w = np.array(control[:8])
        alpha_cmd = control[8]
        beta_cmd = control[9]
        
        alpha = alpha_cmd + self.alpha_offset
        beta = beta_cmd + self.beta_offset
        
        # 扰动
        if disturbance is not None:
            dist_f = np.array(disturbance['force'])
            dist_m = np.array(disturbance['torque'])
        else:
            dist_f = np.zeros(3)
            dist_m = np.zeros(3)
        
        # 位置导数
        px_dot = vx
        py_dot = vy
        pz_dot = vz
        
        # 计算推力
        f = self.Ct * w * np.abs(w)
        
        # 计算反作用力矩
        m = np.zeros(8)
        m[0] = -self.Cd * (w[0] ** 2)
        m[1] = self.Cd * w[1] * np.abs(w[1])
        m[2] = -self.Cd * (w[2] ** 2)
        m[3] = self.Cd * w[3] * np.abs(w[3])
        m[4] = self.Cd * (w[4] ** 2)
        m[5] = -self.Cd * w[5] * np.abs(w[5])
        m[6] = self.Cd * (w[6] ** 2)
        m[7] = -self.Cd * w[7] * np.abs(w[7])
        
        # 机体坐标系力
        fx_b = f[1] * np.sin(alpha) + f[3] * np.sin(beta) + f[5] * np.sin(alpha) + f[7] * np.sin(beta)
        fy_b = 0.0
        fz_b = (f[0] + f[1] * np.cos(alpha) + f[2] + f[3] * np.cos(beta) +
                f[4] + f[5] * np.cos(alpha) + f[6] + f[7] * np.cos(beta))
        
        # 四元数到旋转矩阵
        R00 = 1 - 2 * (q2**2 + q3**2)
        R01 = 2 * (q1*q2 - q0*q3)
        R02 = 2 * (q1*q3 + q0*q2)
        R10 = 2 * (q1*q2 + q0*q3)
        R11 = 1 - 2 * (q1**2 + q3**2)
        R12 = 2 * (q2*q3 - q0*q1)
        R20 = 2 * (q1*q3 - q0*q2)
        R21 = 2 * (q2*q3 + q0*q1)
        R22 = 1 - 2 * (q1**2 + q2**2)
        
        # 转换到世界坐标系
        thrust_accx_w = (R00*fx_b + R01*fy_b + R02*fz_b) / self.mass
        thrust_accy_w = (R10*fx_b + R11*fy_b + R12*fz_b) / self.mass
        thrust_accz_w = (R20*fx_b + R21*fy_b + R22*fz_b) / self.mass
        
        # 速度导数（加入扰动）
        vx_dot = thrust_accx_w + dist_f[0] / self.mass
        vy_dot = thrust_accy_w + dist_f[1] / self.mass
        vz_dot = thrust_accz_w - self.g0 + self.buoyancy / self.mass + dist_f[2] / self.mass
        
        # 四元数导数
        q0_dot = -(q1*wx)/2 - (q2*wy)/2 - (q3*wz)/2
        q1_dot = (q0*wx)/2 - (q3*wy)/2 + (q2*wz)/2
        q2_dot = (q3*wx)/2 + (q0*wy)/2 - (q1*wz)/2
        q3_dot = (q1*wy)/2 - (q2*wx)/2 + (q0*wz)/2
        
        # 机体力矩
        mx = self.l * (f[1] + f[5] - f[3] - f[7])
        my = self.l * self.Ct * (-w[0]**2 - w[4]**2 + w[2]**2 + w[6]**2)
        mz_lr = (-self.l * ((f[1] + f[5]) * np.sin(alpha) - (f[3] + f[7]) * np.sin(beta)) +
                 (m[1] + m[5]) * np.cos(alpha) + (m[3] + m[7]) * np.cos(beta))
        mz_other = (m[0] + m[2] + m[4] + m[6])
        mz = self.k_yaw_lr * mz_lr + self.k_yaw_other * mz_other
        
        # 角速度导数（加入扰动）
        wx_dot = (mx + dist_m[0] + self.Iyy*wy*wz - self.Izz*wy*wz) / self.Ixx
        wy_dot = (my + dist_m[1] - self.Ixx*wx*wz + self.Izz*wx*wz) / self.Iyy
        wz_dot = (mz + dist_m[2] + self.Ixx*wx*wy - self.Iyy*wx*wy) / self.Izz
        
        # 组装状态导数
        state_dot = np.array([
            px_dot, py_dot, pz_dot,
            q0_dot, q1_dot, q2_dot, q3_dot,
            vx_dot, vy_dot, vz_dot,
            wx_dot, wy_dot, wz_dot
        ])
        
        return state_dot
    
    def integrate_rk4(self, state, control, dt, disturbance=None):
        """
        四阶龙格库塔积分
        
        Args:
            state: 当前状态
            control: 控制输入
            dt: 时间步长
            disturbance: 扰动
        
        Returns:
            next_state: 下一时刻状态
        """
        k1 = self.compute_dynamics(state, control, disturbance)
        k2 = self.compute_dynamics(state + 0.5*dt*k1, control, disturbance)
        k3 = self.compute_dynamics(state + 0.5*dt*k2, control, disturbance)
        k4 = self.compute_dynamics(state + dt*k3, control, disturbance)
        
        next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 归一化四元数
        quat = next_state[3:7]
        next_state[3:7] = quat / np.linalg.norm(quat)
        
        return next_state


if __name__ == '__main__':
    # 测试
    dynamics = ForwardDynamics()
    
    # 初始状态：悬停位置
    state = np.array([0, 0, 1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    
    # 悬停控制（8个电机相同转速）
    hover_speed = np.sqrt(dynamics.mass * dynamics.g0 / (8 * dynamics.Ct))
    control = np.array([hover_speed]*8 + [np.pi/2, np.pi/2])
    
    # 无扰动仿真
    print("测试正向动力学模型")
    print(f"初始状态: {state}")
    print(f"悬停转速: {hover_speed:.2f} krpm")
    
    dt = 0.01
    for i in range(100):
        state = dynamics.integrate_rk4(state, control, dt)
    
    print(f"仿真1秒后状态: {state}")
    print(f"位置: [{state[0]:.6f}, {state[1]:.6f}, {state[2]:.6f}]")
    print(f"速度: [{state[7]:.6f}, {state[8]:.6f}, {state[9]:.6f}]")
