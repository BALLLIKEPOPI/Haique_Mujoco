# 轨迹生成器
# 支持悬停、画圆、画方形等轨迹
import numpy as np

class TrajectoryGenerator:
    def __init__(self):
        self.mode = 'hover'  # 'hover', 'circle', 'square', 'climb'
        self.start_time = 0.0
        self.current_time = 0.0
        
        # 悬停参数
        self.hover_position = np.array([0.0, 0.0, 1.0])
        
        # 圆形轨迹参数
        self.circle_center = np.array([0.0, 0.0, 1.0])
        self.circle_radius = 0.5  # 半径0.5m
        self.circle_period = 10.0  # 10秒一圈
        
        # 方形轨迹参数
        self.square_center = np.array([0.0, 0.0, 1.0])
        self.square_size = 0.8  # 边长0.8m
        self.square_period = 16.0  # 16秒一圈（每条边4秒）
        
        # 连续爬升参数
        self.climb_start_height = 0.0  # 起始高度（从地面开始）
        self.climb_target_height = 2.5  # 目标高度
        self.climb_speed = 0.3  # 爬升速度 (m/s)
        
        # 平滑过渡参数
        self.transition_duration = 2.0  # 过渡时间（秒）
        self.transition_start_pos = None  # 过渡起始位置
        self.last_position = np.array([0.0, 0.0, 1.0])  # 上一次的位置
        
    
    def set_mode(self, mode):
        """切换轨迹模式"""
        if mode in ['hover', 'circle', 'square', 'climb']:
            self.mode = mode
            self.start_time = self.current_time
            # 记录过渡起始位置（用于平滑过渡）
            self.transition_start_pos = self.last_position.copy()
            
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
        else:
            print(f"✗ 未知轨迹模式: {mode}")
    
    def get_reference(self, time):
        """根据当前时间和模式生成参考位置"""
        self.current_time = time
        
        if self.mode == 'hover':
            target_pos = self._hover_trajectory()
        elif self.mode == 'circle':
            target_pos = self._circle_trajectory()
        elif self.mode == 'square':
            target_pos = self._square_trajectory()
        elif self.mode == 'climb':
            target_pos = self._climb_trajectory()
        else:
            target_pos = self.hover_position.copy()
        
        # 应用平滑过渡
        final_pos = self._apply_transition(target_pos)
        self.last_position = final_pos.copy()
        
        return final_pos
    
    def _apply_transition(self, target_pos):
        """应用平滑过渡从当前位置到目标轨迹"""
        if self.transition_start_pos is None:
            return target_pos
        
        t = self.current_time - self.start_time
        
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
    
    def _circle_trajectory(self):
        """圆形轨迹"""
        t = self.current_time - self.start_time
        
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
    
    def _square_trajectory(self):
        """方形轨迹"""
        t = self.current_time - self.start_time
        
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
        t = self.current_time - self.start_time
        
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

