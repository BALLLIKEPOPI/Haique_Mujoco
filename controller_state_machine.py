#!/usr/bin/env python3
"""
控制器状态机
根据速度指令自动切换：
- 纯悬停模式：使用aerial控制器（舵机固定为0）
- 水平运动模式：使用underwater控制器（舵机可调）
"""
import numpy as np


class ControllerStateMachine:
    """控制器状态机"""
    
    # 状态定义
    STATE_HOVER = "hover"           # 悬停模式（aerial控制器）
    STATE_HORIZONTAL_MOVE = "move"  # 水平运动模式（underwater控制器）
    
    def __init__(self, 
                 vel_threshold=0.05,       # 速度切换阈值 (m/s)
                 hysteresis_time=0.5,      # 滞后时间，避免频繁切换 (s)
                 dt=0.01):                 # 控制周期 (s)
        """
        初始化状态机
        
        Args:
            vel_threshold: 水平速度阈值，超过此值切换到运动模式
            hysteresis_time: 状态切换滞后时间，避免抖动
            dt: 控制周期
        """
        self.vel_threshold = vel_threshold
        self.hysteresis_samples = int(hysteresis_time / dt)
        
        self.current_state = self.STATE_HOVER
        self.state_counter = 0  # 状态保持计数器
        
        # 状态历史
        self.state_history = []
        
    def update(self, velocity_cmd):
        """
        更新状态机
        
        Args:
            velocity_cmd: 速度指令 [vx, vy, vz, yaw_rate]
            
        Returns:
            current_state: 当前状态 ('hover' or 'move')
            state_changed: 状态是否改变
        """
        vx, vy = velocity_cmd[0], velocity_cmd[1]
        yaw_rate = velocity_cmd[3] if len(velocity_cmd) > 3 else 0.0
        v_horizontal = np.sqrt(vx**2 + vy**2)
        
        # 判断目标状态：有水平移动或偏航控制时使用underwater模式
        if v_horizontal > self.vel_threshold or abs(yaw_rate) > 0.01:
            target_state = self.STATE_HORIZONTAL_MOVE
        else:
            target_state = self.STATE_HOVER
            # target_state = self.STATE_HORIZONTAL_MOVE
        
        state_changed = False
        
        # 状态切换逻辑（带滞后）
        if target_state != self.current_state:
            self.state_counter += 1
            
            # 达到滞后时间，切换状态
            if self.state_counter >= self.hysteresis_samples:
                old_state = self.current_state
                self.current_state = target_state
                self.state_counter = 0
                state_changed = True
                
                print(f"🔄 控制器切换: {old_state} → {self.current_state}")
                if self.current_state == self.STATE_HOVER:
                    print("   使用aerial控制器（舵机固定为0）")
                else:
                    print("   使用underwater控制器（舵机可调）")
        else:
            # 目标状态与当前状态一致，重置计数器
            self.state_counter = 0
        
        # 记录历史
        self.state_history.append(self.current_state)
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
        
        return self.current_state, state_changed
    
    def get_current_state(self):
        """获取当前状态"""
        return self.current_state
    
    def is_hover_mode(self):
        """是否处于悬停模式"""
        return self.current_state == self.STATE_HOVER
    
    def is_move_mode(self):
        """是否处于运动模式"""
        return self.current_state == self.STATE_HORIZONTAL_MOVE
    
    def force_state(self, state):
        """强制设置状态（用于调试）"""
        if state in [self.STATE_HOVER, self.STATE_HORIZONTAL_MOVE]:
            self.current_state = state
            self.state_counter = 0
            print(f"⚠️  强制切换到状态: {state}")
        else:
            print(f"⚠️  无效状态: {state}")
    
    def get_statistics(self):
        """获取状态统计信息"""
        if len(self.state_history) == 0:
            return {"hover_ratio": 0.0, "move_ratio": 0.0}
        
        hover_count = self.state_history.count(self.STATE_HOVER)
        move_count = self.state_history.count(self.STATE_HORIZONTAL_MOVE)
        total = len(self.state_history)
        
        return {
            "hover_ratio": hover_count / total,
            "move_ratio": move_count / total,
            "total_samples": total
        }


def test_state_machine():
    """测试状态机"""
    print("="*60)
    print("🔄 状态机测试")
    print("="*60)
    
    sm = ControllerStateMachine(
        vel_threshold=0.05,
        hysteresis_time=0.5,
        dt=0.01
    )
    
    # 测试场景
    test_scenarios = [
        # (时间, vx, vy, 描述)
        (0.0, 0.0, 0.0, "静止悬停"),
        (1.0, 0.0, 0.0, "保持悬停"),
        (2.0, 0.1, 0.0, "开始前进"),
        (3.0, 0.2, 0.1, "继续运动"),
        (4.0, 0.0, 0.0, "停止运动"),
        (5.0, 0.0, 0.0, "回到悬停"),
        (6.0, 0.0, 0.3, "左移"),
        (7.0, 0.0, 0.0, "再次停止"),
    ]
    
    dt = 0.01
    for t_target, vx, vy, desc in test_scenarios:
        # 每个场景运行1秒
        for step in range(100):
            t = t_target + step * dt
            
            velocity_cmd = np.array([vx, vy, 0.0, 0.0])
            state, changed = sm.update(velocity_cmd)
            
            if changed or step == 0:
                print(f"t={t:.2f}s | {desc:15s} | vx={vx:+.2f} vy={vy:+.2f} | 状态: {state}")
    
    # 统计信息
    stats = sm.get_statistics()
    print(f"\n【统计信息】")
    print(f"  悬停模式占比: {stats['hover_ratio']*100:.1f}%")
    print(f"  运动模式占比: {stats['move_ratio']*100:.1f}%")


if __name__ == '__main__':
    test_state_machine()
