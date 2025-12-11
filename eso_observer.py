"""
扩张状态观测器 (Extended State Observer, ESO)
用于估计无人机受到的未知扰动力和扰动力矩
"""
import numpy as np

class ESO_Observer:
    def __init__(self):
        """
        初始化ESO观测器
        估计6维扰动: [fx, fy, fz, mx, my, mz]
        """
        # 无人机参数
        self.mass = 4.672  # kg
        self.inertia = np.diag([0.10359101, 0.10411340, 0.16095823])  # kg*m^2
        
        # ESO增益参数（需要调优）
        # 线性ESO增益 (位置-速度-扰动力)
        self.beta_pos = np.array([300.0, 300.0, 300.0])  # 位置观测增益
        self.beta_vel = np.array([3000.0, 3000.0, 3000.0])  # 速度观测增益
        self.beta_dist_f = np.array([10000.0, 10000.0, 10000.0])  # 扰动力观测增益
        
        # 角度ESO增益 (姿态-角速度-扰动力矩)
        self.beta_att = np.array([200.0, 200.0, 200.0])  # 姿态观测增益
        self.beta_omega = np.array([2000.0, 2000.0, 2000.0])  # 角速度观测增益
        self.beta_dist_m = np.array([8000.0, 8000.0, 8000.0])  # 扰动力矩观测增益
        
        # ESO状态
        # 线性ESO状态: [位置估计, 速度估计, 扰动力估计]
        self.z_pos = np.zeros(3)  # 位置估计
        self.z_vel = np.zeros(3)  # 速度估计
        self.z_dist_f = np.zeros(3)  # 扰动力估计 [fx, fy, fz]
        
        # 角度ESO状态: [姿态估计(欧拉角), 角速度估计, 扰动力矩估计]
        self.z_att = np.zeros(3)  # 姿态估计 (roll, pitch, yaw)
        self.z_omega = np.zeros(3)  # 角速度估计
        self.z_dist_m = np.zeros(3)  # 扰动力矩估计 [mx, my, mz]
        
        # 低通滤波器参数（平滑扰动估计）
        self.filter_alpha = 0.05  # 滤波系数 (0-1, 越小越平滑)
        self.dist_f_filtered = np.zeros(3)  # 滤波后的扰动力
        self.dist_m_filtered = np.zeros(3)  # 滤波后的扰动力矩
        
        # 扰动饱和限制
        self.max_dist_force = 10.0  # 最大扰动力 (N)
        self.max_dist_torque = 2.0  # 最大扰动力矩 (Nm)
        
        # 初始化标志
        self.initialized = False
        
        # 数据记录
        self.disturbance_log = []
        
    def initialize(self, position, velocity, euler_angles, angular_velocity):
        """
        使用测量值初始化ESO
        
        Args:
            position: [x, y, z] 位置 (m)
            velocity: [vx, vy, vz] 速度 (m/s)
            euler_angles: [roll, pitch, yaw] 欧拉角 (rad)
            angular_velocity: [wx, wy, wz] 角速度 (rad/s)
        """
        self.z_pos = position.copy()
        self.z_vel = velocity.copy()
        self.z_dist_f = np.zeros(3)
        
        self.z_att = euler_angles.copy()
        self.z_omega = angular_velocity.copy()
        self.z_dist_m = np.zeros(3)
        
        self.dist_f_filtered = np.zeros(3)
        self.dist_m_filtered = np.zeros(3)
        
        self.initialized = True
        print("✓ ESO观测器已初始化")
        
    def update(self, position, velocity, euler_angles, angular_velocity, 
               control_force, control_torque, dt):
        """
        更新ESO状态
        
        Args:
            position: [x, y, z] 测量位置 (m)
            velocity: [vx, vy, vz] 测量速度 (m/s)
            euler_angles: [roll, pitch, yaw] 测量姿态 (rad)
            angular_velocity: [wx, wy, wz] 测量角速度 (rad/s)
            control_force: [fx, fy, fz] 控制力 (N) in body frame
            control_torque: [mx, my, mz] 控制力矩 (Nm)
            dt: 时间步长 (s)
        """
        if not self.initialized:
            self.initialize(position, velocity, euler_angles, angular_velocity)
            return
        
        # ========== 线性ESO更新 ==========
        # 位置误差
        e_pos = position - self.z_pos
        
        # ESO微分方程 (离散化欧拉法)
        # z_pos_dot = z_vel + beta_pos * e_pos
        # z_vel_dot = z_dist_f / m + control_force / m + beta_vel * e_pos
        # z_dist_f_dot = beta_dist_f * e_pos
        
        self.z_pos += (self.z_vel + self.beta_pos * e_pos) * dt
        self.z_vel += (self.z_dist_f / self.mass + control_force / self.mass + 
                       self.beta_vel * e_pos) * dt
        self.z_dist_f += self.beta_dist_f * e_pos * dt
        
        # 扰动力饱和限制
        self.z_dist_f = np.clip(self.z_dist_f, -self.max_dist_force, self.max_dist_force)
        
        # ========== 角度ESO更新 ==========
        # 姿态误差
        e_att = euler_angles - self.z_att
        
        # 处理yaw角的周期性 (-pi, pi)
        e_att[2] = np.arctan2(np.sin(e_att[2]), np.cos(e_att[2]))
        
        # ESO微分方程
        # z_att_dot = z_omega + beta_att * e_att
        # z_omega_dot = inv(I) * (z_dist_m + control_torque) + beta_omega * e_att
        # z_dist_m_dot = beta_dist_m * e_att
        
        self.z_att += (self.z_omega + self.beta_att * e_att) * dt
        
        # 简化：假设惯性矩阵对角
        inertia_inv = np.diag(1.0 / np.diag(self.inertia))
        self.z_omega += (inertia_inv @ (self.z_dist_m + control_torque) + 
                        self.beta_omega * e_att) * dt
        self.z_dist_m += self.beta_dist_m * e_att * dt
        
        # 扰动力矩饱和限制
        self.z_dist_m = np.clip(self.z_dist_m, -self.max_dist_torque, self.max_dist_torque)
        
        # ========== 低通滤波 ==========
        self.dist_f_filtered = (self.filter_alpha * self.z_dist_f + 
                               (1 - self.filter_alpha) * self.dist_f_filtered)
        self.dist_m_filtered = (self.filter_alpha * self.z_dist_m + 
                               (1 - self.filter_alpha) * self.dist_m_filtered)
        
        # 记录数据（用于分析）
        self.disturbance_log.append({
            'dist_f': self.z_dist_f.copy(),
            'dist_m': self.z_dist_m.copy(),
            'dist_f_filtered': self.dist_f_filtered.copy(),
            'dist_m_filtered': self.dist_m_filtered.copy()
        })
        
    def get_disturbance_force(self, filtered=True):
        """
        获取估计的扰动力
        
        Args:
            filtered: 是否返回滤波后的值
            
        Returns:
            [fx, fy, fz] 扰动力 (N) in body frame
        """
        if filtered:
            return self.dist_f_filtered.copy()
        else:
            return self.z_dist_f.copy()
    
    def get_disturbance_torque(self, filtered=True):
        """
        获取估计的扰动力矩
        
        Args:
            filtered: 是否返回滤波后的值
            
        Returns:
            [mx, my, mz] 扰动力矩 (Nm)
        """
        if filtered:
            return self.dist_m_filtered.copy()
        else:
            return self.z_dist_m.copy()
    
    def get_all_disturbances(self, filtered=True):
        """
        获取所有扰动估计
        
        Returns:
            dict: {'force': [fx,fy,fz], 'torque': [mx,my,mz]}
        """
        return {
            'force': self.get_disturbance_force(filtered),
            'torque': self.get_disturbance_torque(filtered)
        }
    
    def reset(self):
        """重置ESO观测器"""
        self.z_pos = np.zeros(3)
        self.z_vel = np.zeros(3)
        self.z_dist_f = np.zeros(3)
        self.z_att = np.zeros(3)
        self.z_omega = np.zeros(3)
        self.z_dist_m = np.zeros(3)
        self.dist_f_filtered = np.zeros(3)
        self.dist_m_filtered = np.zeros(3)
        self.initialized = False
        self.disturbance_log = []
        
    def set_gains(self, beta_pos=None, beta_vel=None, beta_dist_f=None,
                  beta_att=None, beta_omega=None, beta_dist_m=None):
        """
        设置ESO增益参数
        
        增益越大，观测器响应越快，但可能更容易受噪声影响
        """
        if beta_pos is not None:
            self.beta_pos = np.array(beta_pos)
        if beta_vel is not None:
            self.beta_vel = np.array(beta_vel)
        if beta_dist_f is not None:
            self.beta_dist_f = np.array(beta_dist_f)
        if beta_att is not None:
            self.beta_att = np.array(beta_att)
        if beta_omega is not None:
            self.beta_omega = np.array(beta_omega)
        if beta_dist_m is not None:
            self.beta_dist_m = np.array(beta_dist_m)
            
        print("✓ ESO增益参数已更新")
    
    def save_disturbance_log(self, filename='eso_disturbance_log.csv'):
        """保存扰动估计日志"""
        import csv
        
        if len(self.disturbance_log) == 0:
            print("⚠️  没有数据可保存")
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['dist_fx', 'dist_fy', 'dist_fz', 
                         'dist_mx', 'dist_my', 'dist_mz',
                         'dist_fx_filt', 'dist_fy_filt', 'dist_fz_filt',
                         'dist_mx_filt', 'dist_my_filt', 'dist_mz_filt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for log in self.disturbance_log:
                writer.writerow({
                    'dist_fx': log['dist_f'][0],
                    'dist_fy': log['dist_f'][1],
                    'dist_fz': log['dist_f'][2],
                    'dist_mx': log['dist_m'][0],
                    'dist_my': log['dist_m'][1],
                    'dist_mz': log['dist_m'][2],
                    'dist_fx_filt': log['dist_f_filtered'][0],
                    'dist_fy_filt': log['dist_f_filtered'][1],
                    'dist_fz_filt': log['dist_f_filtered'][2],
                    'dist_mx_filt': log['dist_m_filtered'][0],
                    'dist_my_filt': log['dist_m_filtered'][1],
                    'dist_mz_filt': log['dist_m_filtered'][2],
                })
        
        print(f"✓ 扰动估计日志已保存到 {filename}")

