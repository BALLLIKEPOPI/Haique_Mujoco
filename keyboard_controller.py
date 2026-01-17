#!/usr/bin/env python3
"""
键盘控制节点 - WSL手柄不可用时的替代方案
使用pygame键盘输入模拟手柄
"""
import pygame
import numpy as np
import time
import threading


class KeyboardController:
    """键盘控制器（模拟手柄）"""
    
    def __init__(self, 
                 max_vel_xy=1.0,
                 max_vel_z=0.8,
                 max_yaw_rate=1.0,
                 accel_rate=2.0):
        """
        初始化键盘控制器
        
        Args:
            max_vel_xy: 最大水平速度 (m/s)
            max_vel_z: 最大垂直速度 (m/s)
            max_yaw_rate: 最大偏航角速度 (rad/s)
            accel_rate: 加速度 (m/s²)
        """
        self.max_vel_xy = max_vel_xy
        self.max_vel_z = max_vel_z
        self.max_yaw_rate = max_yaw_rate
        self.accel_rate = accel_rate
        
        # 当前速度指令 [vx, vy, vz, yaw_rate]
        self.velocity_cmd = np.zeros(4)
        self.target_velocity = np.zeros(4)
        
        self.is_active = False
        self.emergency_stop = False
        
        # 初始化pygame
        pygame.init()
        pygame.display.set_mode((400, 300))
        pygame.display.set_caption("键盘控制 - 按H查看帮助")
        
        self.running = False
        self.thread = None
        
        self.show_help = True
        
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("⌨️  键盘控制说明")
        print("="*60)
        print("  W/S     - 前进/后退")
        print("  A/D     - 左移/右移")
        print("  空格    - 上升")
        print("  Shift   - 下降")
        print("  Q/E     - 偏航左/右")
        print("  ESC     - 紧急停止")
        print("  R       - 解除紧急停止")
        print("  H       - 显示此帮助")
        print("="*60 + "\n")
    
    def _update_loop(self):
        """键盘更新循环"""
        clock = pygame.time.Clock()
        dt = 0.02  # 50Hz
        
        if self.show_help:
            self._print_help()
        
        while self.running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_h:
                        self._print_help()
                    elif event.key == pygame.K_ESCAPE:
                        self.emergency_stop = True
                        self.target_velocity = np.zeros(4)
                        print("🛑 紧急停止！按R解除")
                    elif event.key == pygame.K_r:
                        if self.emergency_stop:
                            self.emergency_stop = False
                            print("✓ 紧急停止已解除")
            
            if self.emergency_stop:
                self.velocity_cmd = np.zeros(4)
                clock.tick(50)
                continue
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 目标速度
            target_vx = 0.0
            target_vy = 0.0
            target_vz = 0.0
            target_yaw_rate = 0.0
            
            # W - 前进（移除S后退）
            if keys[pygame.K_w]:
                target_vx = self.max_vel_xy
            
            # 空格/Shift - 垂直移动
            if keys[pygame.K_SPACE]:
                target_vz = self.max_vel_z
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                target_vz = -self.max_vel_z
            
            # A/D - 偏航控制（左转/右转）
            if keys[pygame.K_a]:
                target_yaw_rate = self.max_yaw_rate  # 左转（修正方向）
            if keys[pygame.K_d]:
                target_yaw_rate = -self.max_yaw_rate  # 右转（修正方向）
            
            # Q/E - 备用偏航（保留兼容性）
            if keys[pygame.K_q]:
                target_yaw_rate = -self.max_yaw_rate
            if keys[pygame.K_e]:
                target_yaw_rate = self.max_yaw_rate
            
            self.target_velocity = np.array([target_vx, target_vy, target_vz, target_yaw_rate])
            
            # 平滑加速
            diff = self.target_velocity - self.velocity_cmd
            max_change = self.accel_rate * dt
            
            for i in range(4):
                if abs(diff[i]) > max_change:
                    self.velocity_cmd[i] += np.sign(diff[i]) * max_change
                else:
                    self.velocity_cmd[i] = self.target_velocity[i]
            
            # 检查是否有输入
            self.is_active = np.any(np.abs(self.velocity_cmd) > 1e-3)
            
            clock.tick(50)
    
    def start(self):
        """启动键盘控制"""
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        print("✅ 键盘控制已启动")
        return True
    
    def stop(self):
        """停止键盘控制"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        pygame.quit()
        print("✓ 键盘控制已停止")
    
    def get_velocity_cmd(self):
        """获取速度指令"""
        return self.velocity_cmd.copy(), self.is_active
    
    def is_emergency_stopped(self):
        """检查紧急停止状态"""
        return self.emergency_stop


def test_keyboard():
    """测试键盘控制"""
    print("="*60)
    print("⌨️  键盘控制测试")
    print("="*60)
    
    kbd = KeyboardController()
    kbd.start()
    
    print("\n按键测试运行中，按Ctrl+C退出\n")
    
    try:
        while True:
            vel_cmd, is_active = kbd.get_velocity_cmd()
            
            if is_active:
                print(f"\r速度: vx={vel_cmd[0]:+.2f} vy={vel_cmd[1]:+.2f} "
                      f"vz={vel_cmd[2]:+.2f} yaw={vel_cmd[3]:+.2f}  ",
                      end='', flush=True)
            else:
                print(f"\r等待按键输入...                                    ",
                      end='', flush=True)
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n退出测试...")
    finally:
        kbd.stop()


if __name__ == '__main__':
    test_keyboard()
