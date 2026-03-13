"""
Underwater PID Controller for Haique Drone (Final Corrected Version)
构型: 8电机 + 2舵机 (Puffin)
关键定义: 舵机 0度 = 平行于 X 轴 (水平向前)
分配逻辑:
- Heave (升力): 仅由 前后电机 (Fixed Group) 承担
- Surge (前进): 仅由 左右电机 (Tilt Group) 承担
- Roll  (横滚): 左右舵机 差速倾转 (Diff Tilt)
- Yaw   (偏航): 左右电机 差速推力 (Diff Thrust)
- Pitch (俯仰): 前后电机 差速推力
"""

import numpy as np
import os
import csv
import time
from typing import Tuple, Optional
from scipy.spatial.transform import Rotation as R

try:
    from model.config_loader import get_mode_config, get_value
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

class PIDControllerUnderwater:
    def __init__(self, config_path: str = None, dt: float = 0.01):
        self.dt = dt
        self.start_time = None
        
        # 1. 加载物理参数
        if HAS_CONFIG and config_path:
            cfg = get_mode_config("underwater", path=config_path)
            self.mass = float(get_value(cfg, "physical.mass", 4.672))
            self.g = float(get_value(cfg, "physical.g", 9.8066))
            self.max_thrust = float(get_value(cfg, "sim.max_thrust", 15.0))
            self.left_offset = float(get_value(cfg, "servo.left_offset", -0.1))
            self.right_offset = float(get_value(cfg, "servo.right_offset", 0.0))
        else:
            self.mass = 4.672
            self.g = 9.8066
            self.max_thrust = 15.0
            self.left_offset = -0.1
            self.right_offset = 0.0

        print(f"PID Init: Mass={self.mass:.3f}, G={self.g:.3f} (0 deg = Horizontal)")

        # --- PID 参数 (串级结构) ---
        
        # [位置环] (Position -> Accel)
        # Z轴积分 (Ki) 至关重要，因为只有4个电机提供升力，模型误差可能较大
        self.kp_pos = np.array([18.0, 15.0, 50.0])  
        self.ki_pos = np.array([2.0, 2.0, 15.0])   
        self.kd_pos = np.array([5.0, 4.5, 80.0])

        # [姿态环] (Angle -> Rate)
        self.kp_att = np.array([6.0, 12.0, 10.0])  # Roll↓(舵机慢), Pitch保持, Yaw↑(需更强保持)

        # [角速度环] (Rate -> Torque) - 刚度层
        # Roll (靠舵机): 响应可能稍慢，给大一点的 P
        # Yaw (靠推力): 响应快
        self.kp_rate = np.array([20.0, 25.0, 20.0])  # Roll↓(减少超调), Pitch保持, Yaw↑(更快响应)
        self.ki_rate = np.array([0.5, 0.2, 0.8])  # Roll↓, Pitch保持, Yaw↑(抵消漂移)
        self.kd_rate = np.array([6.0, 2.0, 2.5])  # Roll↑↑(增加阻尼), Pitch保持, Yaw↑

        self.int_err_pos = np.zeros(3)
        self.int_err_rate = np.zeros(3)
        
        # 电机输出低通滤波器
        self.motor_output_filtered = np.zeros(8)  # u0-u7的滤波状态
        self.motor_filter_tau = 0.02  # 时间常数50ms (截止频率约3.2Hz)
        
        self.data_log = []

    def run(self, current_state: np.ndarray, target_state: np.ndarray, dt: float = None):
        if dt is None: dt = self.dt
        if self.start_time is None: self.start_time = time.time()
        
        # 状态解包
        curr_pos = current_state[0:3]
        curr_quat = current_state[3:7] 
        curr_vel = current_state[7:10] 
        curr_omega = current_state[10:13] 
        
        target_pos = target_state[0:3]
        target_vel = target_state[7:10] if len(target_state) >= 10 else np.zeros(3)  # 提取目标速度
        vel_magnitude = np.linalg.norm(target_vel[:2])
        if vel_magnitude > 0.1:  # 速度>0.1m/s时朝运动方向
            target_yaw = np.arctan2(target_vel[1], target_vel[0])
        else:
            target_yaw = 0.0  # 静止时保持0度
        print(target_yaw)
        
        r = R.from_quat([curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]])
        R_matrix = r.as_matrix()
        
        # --- 1. 位置控制（含速度前馈） ---
        pos_err = target_pos - curr_pos
        vel_err = target_vel - curr_vel  # 速度误差
        
        # Z轴积分抗饱和
        if curr_pos[2] > 0.1 or abs(pos_err[2]) < 0.5:
            self.int_err_pos += pos_err * dt
        self.int_err_pos = np.clip(self.int_err_pos, -4.0, 4.0) # 允许更大的积分修正
        
        # PID控制 + 速度前馈
        acc_des = (self.kp_pos * pos_err) + (self.kd_pos * vel_err) + (self.ki_pos * self.int_err_pos)
        acc_des[2] += self.g 
        
        force_world = self.mass * acc_des
        force_body = R_matrix.T @ force_world
        
        # --- 2. 姿态控制 ---
        curr_rpy = r.as_euler('xyz')
        target_rpy = np.array([0.0, 0.0, target_yaw])
        
        att_err = target_rpy - curr_rpy
        for i in range(3):
            if att_err[i] > np.pi: att_err[i] -= 2*np.pi
            if att_err[i] < -np.pi: att_err[i] += 2*np.pi
            
        # Roll死区: ±2°
        roll_deadzone = np.deg2rad(1.0)  # 0.0349 rad
        for i in range(3):
            if abs(att_err[i]) < roll_deadzone:
                att_err[i] = 0.0

        rate_des = np.clip(self.kp_att * att_err, -2.5, 2.5)
        rate_err = rate_des - curr_omega
        
        self.int_err_rate += rate_err * dt
        self.int_err_rate = np.clip(self.int_err_rate, -2.0, 2.0)
        
        torque_des = (self.kp_rate * rate_err) + (self.ki_rate * self.int_err_rate)
        
        # --- 3. 混控分配 (0度=水平 专用版) ---
        u = self.control_allocation(force_body, torque_des)
        
        self.log_step(current_state, target_state, u, force_body, torque_des)
        return u

    def control_allocation(self, force, torque):
        """
        Force: [Fx(Surge), Fy, Fz(Heave)]
        Torque: [Tx(Roll), Ty(Pitch), Tz(Yaw)]
        """
        Fx, Fy, Fz = force
        Tx, Ty, Tz = torque
        
        # 1. 升力 (Heave) -> 仅分配给 前后电机 [0,2,4,6]
        # 左右电机水平，无法提供升力
        # 4个电机分担全部重量
        u_heave = Fz / 4.0 
        
        # 2. 前进 (Surge) -> 仅分配给 左右电机 [1,3,5,7]
        # 左右电机水平，效率最高
        u_surge = Fx / 4.0
        
        # 3. 力矩分配
        
        # [Pitch Ty] -> 前后差速
        u_pitch = Ty / (2.0 * 0.3)
        
        # [Yaw Tz] -> 左右差速推力 (因为0度是水平的，推力差产生Yaw)
        # Left(1,5) at +Y, Right(3,7) at -Y
        # Force along +X.
        # Torque = r x F. Left: (+y) x (+x) = -z (Negative Yaw). Right: (-y) x (+x) = +z (Positive Yaw).
        # Tz > 0 (Yaw Left) => Need Right Thrust > Left Thrust
        u_yaw = Tz / (2.0 * 0.3)
        
        # [Roll Tx] -> 左右差速倾转 (舵机主导) + 推力辅助
        # 倾转电机的推力差也能辅助Roll (力矩臂 ~0.3m)
        u_roll = abs(Tx / (2.0 * 0.3))
        # print(u_surge, u_yaw, u_roll)
        
        # 舵机角度控制 (主要Roll控制)
        # 0度是水平。
        # 左舵机向上转 (+alpha) -> 产生 +Z 分力 -> +Roll (Right Roll? depends on axis)
        # 假设 +Tx 代表向右滚 (右翼下沉，左翼上升)
        # Need Left Lift Up, Right Lift Down.
        # Left Servo: +alpha (Up). Right Servo: -alpha (Down).
        # 增益需要大，因为 tilt 产生的垂直分力是 F*sin(alpha)
        roll_gain = -0.04 # 映射力矩到角度 (负号: Tx>0右滚→左舵机减小角度)
        servo_roll_cmd = np.clip(Tx * roll_gain, -0.5, 0.5)
        
        # --- 组装 ---
        u = np.zeros(10)
        
        # Fixed Group (Front/Rear): Heave + Pitch
        u[0] = u_heave - u_pitch
        u[4] = u_heave - u_pitch
        u[2] = u_heave + u_pitch
        u[6] = u_heave + u_pitch
        
        # # Tilt Group (Left/Right): Surge + Yaw + Roll辅助
        # # Left: Surge - Yaw + Roll (增加推力辅助左滚)
        # u[1] = u_surge - 0.5 * u_yaw + 0.3 * u_roll
        # u[5] = u_surge - 0.5 * u_yaw + 0.3 * u_roll
        # # Right: Surge + 0.5 * Yaw - 0.3 * Roll (减少推力辅助右滚)
        # u[3] = u_surge + 0.5 * u_yaw + 0.3 * u_roll
        # u[7] = u_surge + 0.5 * u_yaw + 0.3 * u_roll
        u_surge = np.clip(u_surge, 0, self.max_thrust/4.0)  # Surge推力限制

        u[1] = 0.5 * u_roll - 0.1 * u_yaw + 0.3 * u_surge
        u[5] = 0.5 * u_roll - 0.1 * u_yaw + 0.3 * u_surge
        # Righ 3 5 Roll (减少推力0.3 * 
        u[3] = 0.5 * u_roll + 0.1 * u_yaw + 0.3 * u_surge
        u[7] = 0.5 * u_roll + 0.1 * u_yaw + 0.3 * u_surge
        print(u_surge, u_yaw, u_roll)
        
        u[:8] = np.clip(u[:8], 0.0, self.max_thrust)
        
        # 电机输出低通滤波 (一阶滤波器)
        alpha = self.dt / (self.dt + self.motor_filter_tau)
        self.motor_output_filtered = alpha * u[:8] + (1 - alpha) * self.motor_output_filtered
        u[:8] = self.motor_output_filtered
        
        # # Servos: Roll control via Tilt
        # Left: +gamma, Right: -gamma
        base_tilt = 1.57 
        u[8] = (base_tilt + servo_roll_cmd) - self.left_offset
        u[9] = (base_tilt - servo_roll_cmd) - self.right_offset
        u[8:] = np.clip(u[8:], 1.07, 2.07)
        
        return u

    def log_step(self, state, target, u, force, torque):
        self.data_log.append({
            'time': time.time(),
            'state': state.tolist(),
            'goal': target.tolist(),
            'control': u.tolist(),
            'des_force_body': force.tolist(),
            'des_torque_body': torque.tolist()
        })

    def save_data(self, filename):
        if not self.data_log: return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 表头：完整状态 + 目标位置 + 控制输出
            header = ['time',
                     'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz',
                     'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                     'goal_x', 'goal_y', 'goal_z',
                     'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10']
            writer.writerow(header)
            
            t0 = self.data_log[0]['time']
            for d in self.data_log:
                row = [d['time'] - t0]  # 相对时间
                row.extend(d['state'][:13])  # 13维状态: px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz
                row.extend(d['goal'][:3])    # 目标位置: goal_x, goal_y, goal_z
                row.extend(d['control'])     # 10维控制: u1~u8(电机), u9,u10(舵机)
                writer.writerow(row)
        print(f"✓ PID数据已保存到: {filename} ({len(self.data_log)} 条记录)")