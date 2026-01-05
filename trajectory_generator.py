# 轨迹生成器
# 支持悬停、画圆、画方形等轨迹
import numpy as np

class TrajectoryGenerator:
    def __init__(self):
        self.mode = 'hover'  # 'hover', 'circle', 'square', 'climb', 'forward'
        self.start_time = 0.0
        self.current_time = 0.0

        # 非悬停轨迹的“预悬停”时间：先在 hover_position 稳住，再开始平滑过渡+跟踪轨迹
        self.pre_hover_duration = 2.0  # s
        self.motion_start_time = 0.0
        
        # 悬停参数
        self.hover_position = np.array([0.0, 0.0, 1.0])
        
        # 圆形轨迹参数
        self.circle_center = np.array([0.0, 0.0, 1.0])
        self.circle_radius = 1.0  # 半径1.0m
        self.circle_period = 20.0  # 20秒一圈
        
        # 方形轨迹参数
        self.square_center = np.array([0.0, 0.0, 1.0])
        self.square_size = 3  # 边长3m
        self.square_period = 25.0  # 25秒一圈（每条边6.25秒）
        
        # 连续爬升参数
        self.climb_start_height = 0.0  # 起始高度（从地面开始）
        self.climb_target_height = 2.5  # 目标高度
        self.climb_speed = 0.3  # 爬升速度 (m/s)

        # 匀速前进参数（沿 +X 方向）
        self.forward_speed = 0.5  # m/s
        self.forward_start_pos = self.hover_position.copy()
        
        # 平滑过渡参数
        self.transition_duration = 2.0  # 过渡时间（秒）
        self.transition_start_pos = None  # 过渡起始位置
        self.last_position = np.array([0.0, 0.0, 1.0])  # 上一次的位置
        
    
    def set_mode(self, mode):
        """切换轨迹模式"""
        if mode in ['hover', 'circle', 'square', 'climb', 'forward']:
            self.mode = mode
            self.start_time = self.current_time

            if mode == 'forward':
                # forward 轨迹从悬停点出发
                self.forward_start_pos = self.hover_position.copy()

            # 两阶段：先悬停，再开始过渡/运动
            if mode == 'hover':
                self.motion_start_time = self.current_time
                # 悬停模式：从当前位置（上一参考）平滑过渡到 hover_position
                self.transition_start_pos = self.last_position.copy()
            else:
                self.motion_start_time = self.current_time + self.pre_hover_duration
                # 非悬停轨迹：预悬停期间参考固定在 hover_position；真正开始运动时从 hover_position 过渡
                self.transition_start_pos = self.hover_position.copy()
            
            print(f"✓ 切换轨迹模式: {mode}")
            if mode == 'hover':
                print(f"  目标位置: {self.hover_position}")
            elif mode == 'circle':
                print(f"  圆心: {self.circle_center}, 半径: {self.circle_radius}m, 周期: {self.circle_period}s")
                print(f"  → 将平滑过渡到轨迹起点 (耗时{self.transition_duration}s)")
            elif mode == 'square':
                print(f"  中心: {self.square_center}, 边长: {self.square_size}m, 周期: {self.square_period}s")
                print(f"  → 将平滑过渡到轨迹起点 (耗时{self.transition_duration}s)")
            elif mode == 'climb':
                height_diff = self.climb_target_height - self.climb_start_height
                climb_time = abs(height_diff) / self.climb_speed
                print(f"  连续爬升: {self.climb_start_height}m → {self.climb_target_height}m")
                print(f"  爬升速度: {self.climb_speed} m/s")
                print(f"  预计时间: {climb_time:.1f} 秒")
            elif mode == 'forward':
                print(f"  匀速前进: speed={self.forward_speed} m/s (沿 +X)")
                print(f"  → 将平滑过渡到起点 (耗时{self.transition_duration}s)")
        else:
            print(f"✗ 未知轨迹模式: {mode}")
    
    def get_reference(self, time):
        """根据当前时间和模式生成参考位置"""
        self.current_time = time

        # 预悬停：非 hover 模式先保持悬停参考
        if (self.mode != 'hover') and (self.current_time < self.motion_start_time):
            final_pos = self.hover_position.copy()
            self.last_position = final_pos.copy()
            return final_pos
        
        if self.mode == 'hover':
            target_pos = self._hover_trajectory()
        elif self.mode == 'circle':
            target_pos = self._circle_trajectory()
        elif self.mode == 'square':
            target_pos = self._square_trajectory()
        elif self.mode == 'climb':
            target_pos = self._climb_trajectory()
        elif self.mode == 'forward':
            target_pos = self._forward_trajectory()
        else:
            target_pos = self.hover_position.copy()
        
        # 应用平滑过渡
        final_pos = self._apply_transition(target_pos)
        self.last_position = final_pos.copy()
        
        return final_pos

    def get_reference_state(self, time):
        """根据当前时间和模式生成参考状态（位置+速度）。

        返回:
            pos: np.ndarray shape (3,)
            vel: np.ndarray shape (3,)
        """
        self.current_time = time

        # 预悬停：非 hover 模式先保持悬停参考（位置+速度）
        if (self.mode != 'hover') and (self.current_time < self.motion_start_time):
            final_pos = self.hover_position.copy()
            final_vel = np.zeros(3)
            self.last_position = final_pos.copy()
            return final_pos, final_vel

        if self.mode == 'hover':
            target_pos = self._hover_trajectory()
            target_vel = np.zeros(3)
        elif self.mode == 'circle':
            target_pos, target_vel = self._circle_trajectory_state()
        elif self.mode == 'square':
            target_pos, target_vel = self._square_trajectory_state()
        elif self.mode == 'climb':
            target_pos, target_vel = self._climb_trajectory_state()
        elif self.mode == 'forward':
            target_pos, target_vel = self._forward_trajectory_state()
        else:
            target_pos = self.hover_position.copy()
            target_vel = np.zeros(3)

        # 过渡期位置用插值；速度前馈在过渡期置零，避免目标速度跳变
        final_pos = self._apply_transition(target_pos)
        if self._elapsed_motion_time() < self.transition_duration:
            final_vel = np.zeros(3)
        else:
            final_vel = target_vel

        self.last_position = final_pos.copy()
        return final_pos, final_vel
    
    def _apply_transition(self, target_pos):
        """应用平滑过渡从当前位置到目标轨迹"""
        if self.transition_start_pos is None:
            return target_pos

        t = self._elapsed_motion_time()
        
        # 在过渡时间内，进行插值
        if t < self.transition_duration:
            # 使用平滑的S曲线插值（ease-in-out）
            progress = t / self.transition_duration
            # 三次平滑函数: 3t^2 - 2t^3
            smooth_progress = 3 * progress**2 - 2 * progress**3
            
            interpolated_pos = (1 - smooth_progress) * self.transition_start_pos + smooth_progress * target_pos
            return interpolated_pos
        else:
            # 过渡完成，返回目标轨迹
            return target_pos
    
    def _hover_trajectory(self):
        """悬停轨迹"""
        return self.hover_position.copy()

    def _elapsed_motion_time(self):
        """返回进入“运动阶段”后的时间（扣除预悬停）。"""
        return max(0.0, float(self.current_time - self.motion_start_time))
    
    def _circle_trajectory(self):
        """圆形轨迹"""
        t = self._elapsed_motion_time()
        
        # 在过渡期间，返回轨迹起点（避免追逐移动目标）
        if t < self.transition_duration:
            theta = 0  # 起点：圆的最右侧
        else:
            # 过渡完成后，从起点开始画圆
            t_actual = t - self.transition_duration
            theta = 2 * np.pi * t_actual / self.circle_period
        
        x = self.circle_center[0] + self.circle_radius * np.cos(theta)
        y = self.circle_center[1] + self.circle_radius * np.sin(theta)
        z = self.circle_center[2]
        
        return np.array([x, y, z])

    def _forward_trajectory(self):
        """匀速前进轨迹（沿 +X）"""
        t = self._elapsed_motion_time()

        # 过渡期间固定在起点，避免追逐移动目标
        if t < self.transition_duration:
            return self.forward_start_pos.copy()

        t_actual = t - self.transition_duration
        x = self.forward_start_pos[0] + self.forward_speed * t_actual
        y = self.forward_start_pos[1]
        z = self.forward_start_pos[2]
        return np.array([x, y, z])

    def _forward_trajectory_state(self):
        """匀速前进轨迹（位置+速度）"""
        t = self._elapsed_motion_time()
        pos = self._forward_trajectory()
        if t < self.transition_duration:
            vel = np.zeros(3)
        else:
            vel = np.array([self.forward_speed, 0.0, 0.0])
        return pos, vel

    def _circle_trajectory_state(self):
        """圆形轨迹（位置+解析速度）"""
        t = self._elapsed_motion_time()

        if t < self.transition_duration:
            theta = 0.0
            omega = 0.0
        else:
            t_actual = t - self.transition_duration
            omega = 2 * np.pi / self.circle_period
            theta = omega * t_actual

        x = self.circle_center[0] + self.circle_radius * np.cos(theta)
        y = self.circle_center[1] + self.circle_radius * np.sin(theta)
        z = self.circle_center[2]

        vx = -self.circle_radius * omega * np.sin(theta)
        vy =  self.circle_radius * omega * np.cos(theta)
        vz = 0.0

        return np.array([x, y, z]), np.array([vx, vy, vz])

    def _square_trajectory_state(self):
        """方形轨迹（位置+近似速度）。"""
        # 为了保持简单：过渡期速度置零；过渡完成后按分段常速度给前馈。
        t = self._elapsed_motion_time()
        pos = self._square_trajectory()
        if t < self.transition_duration:
            return pos, np.zeros(3)

        half_size = self.square_size / 2.0
        t_actual = t - self.transition_duration
        seg_time = self.square_period / 4.0
        vmag = (2.0 * half_size) / seg_time

        t_norm = (t_actual % self.square_period)
        seg = int(t_norm // seg_time)

        if seg == 0:      # 右边：y 递增
            vel = np.array([0.0, vmag, 0.0])
        elif seg == 1:    # 上边：x 递减
            vel = np.array([-vmag, 0.0, 0.0])
        elif seg == 2:    # 左边：y 递减
            vel = np.array([0.0, -vmag, 0.0])
        else:             # 下边：x 递增
            vel = np.array([vmag, 0.0, 0.0])

        return pos, vel

    def _climb_trajectory_state(self):
        """爬升轨迹（位置+速度）"""
        t = self._elapsed_motion_time()
        pos = self._climb_trajectory()

        height_diff = self.climb_target_height - self.climb_start_height
        total_climb_time = abs(height_diff) / self.climb_speed
        if 0.0 < t < total_climb_time:
            vz = np.sign(height_diff) * self.climb_speed
        else:
            vz = 0.0
        return pos, np.array([0.0, 0.0, vz])
    
    def _square_trajectory(self):
        """方形轨迹"""
        t = self._elapsed_motion_time()
        
        half_size = self.square_size / 2.0
        cx, cy, cz = self.square_center
        
        # 在过渡期间，返回轨迹起点（右下角）
        if t < self.transition_duration:
            x = cx + half_size
            y = cy - half_size
        else:
            # 过渡完成后，从起点开始画方形
            t_actual = t - self.transition_duration
            # 归一化时间 [0, 1)
            t_norm = (t_actual % self.square_period) / self.square_period
            
            # 分成4段：右→上→左→下
            if t_norm < 0.25:  # 右边：从下到上
                progress = t_norm / 0.25
                x = cx + half_size
                y = cy + (2 * progress - 1) * half_size
            elif t_norm < 0.5:  # 上边：从右到左
                progress = (t_norm - 0.25) / 0.25
                x = cx + (1 - 2 * progress) * half_size
                y = cy + half_size
            elif t_norm < 0.75:  # 左边：从上到下
                progress = (t_norm - 0.5) / 0.25
                x = cx - half_size
                y = cy + (1 - 2 * progress) * half_size
            else:  # 下边：从左到右
                progress = (t_norm - 0.75) / 0.25
                x = cx + (2 * progress - 1) * half_size
                y = cy - half_size
        
        z = cz
        
        return np.array([x, y, z])
    
    def _climb_trajectory(self):
        """连续线性爬升轨迹"""
        t = self._elapsed_motion_time()
        
        # 计算总爬升时间
        height_diff = self.climb_target_height - self.climb_start_height
        total_climb_time = abs(height_diff) / self.climb_speed
        
        if t <= 0:
            # 还未开始爬升
            z = self.climb_start_height
        elif t >= total_climb_time:
            # 已经到达目标高度，保持悬停
            z = self.climb_target_height
        else:
            # 爬升过程中，线性插值
            progress = t / total_climb_time
            z = self.climb_start_height + height_diff * progress
        
        return np.array([0.0, 0.0, z])
    
    def set_hover_position(self, position):
        """设置悬停位置"""
        self.hover_position = np.array(position)
        if self.mode == 'hover':
            print(f"✓ 更新悬停位置: {self.hover_position}")
    
    def set_circle_params(self, center=None, radius=None, period=None):
        """设置圆形轨迹参数"""
        if center is not None:
            self.circle_center = np.array(center)
        if radius is not None:
            self.circle_radius = radius
        if period is not None:
            self.circle_period = period
        print(f"✓ 圆形轨迹参数: 中心={self.circle_center}, 半径={self.circle_radius}m, 周期={self.circle_period}s")
    
    def set_square_params(self, center=None, size=None, period=None):
        """设置方形轨迹参数"""
        if center is not None:
            self.square_center = np.array(center)
        if size is not None:
            self.square_size = size
        if period is not None:
            self.square_period = period
        print(f"✓ 方形轨迹参数: 中心={self.square_center}, 边长={self.square_size}m, 周期={self.square_period}s")
    
    def set_climb_params(self, start_height=None, target_height=None, speed=None):
        """设置连续爬升参数"""
        if start_height is not None:
            self.climb_start_height = start_height
        if target_height is not None:
            self.climb_target_height = target_height
        if speed is not None:
            self.climb_speed = speed
        
        height_diff = self.climb_target_height - self.climb_start_height
        climb_time = abs(height_diff) / self.climb_speed
        
        print(f"✓ 连续爬升参数: {self.climb_start_height}m → {self.climb_target_height}m")
        print(f"  爬升速度: {self.climb_speed} m/s")
        print(f"  预计时间: {climb_time:.1f} 秒")

    def set_forward_params(self, speed=None):
        """设置匀速前进参数"""
        if speed is not None:
            self.forward_speed = float(speed)
        print(f"✓ 匀速前进参数: speed={self.forward_speed} m/s (沿 +X)")

