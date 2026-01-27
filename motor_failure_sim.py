#!/usr/bin/env python3
"""
电机失能仿真脚本
模拟无人机在悬停时某个电机失效的情况
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from os.path import dirname, join, abspath

from control.nmpc_controller import NMPC_Controller
from trajectory_generator import TrajectoryGenerator
from model.config_loader import get_mode_config, get_value

np.set_printoptions(suppress=True, precision=4)


class MotorFailureSimulator:
    def __init__(self, scene_path, config_path=None):
        """初始化电机失能仿真器
        
        Args:
            scene_path: MuJoCo场景XML文件路径
            config_path: 配置文件路径（可选）
        """
        if config_path is None:
            config_path = join(dirname(abspath(__file__)), "config.yaml")
        
        # 加载配置
        cfg = get_mode_config("aerial", path=config_path)
        self.dt = float(get_value(cfg, "physical.dt", 0.005))
        
        # 物理参数
        self.g0 = float(get_value(cfg, "physical.g", 9.8066))
        self.mass = float(get_value(cfg, "physical.mass", 4.672))
        self.Ct = float(get_value(cfg, "physical.Ct", 0.1757))
        self.Cd = float(get_value(cfg, "physical.Cd", 0.02))
        self.dq = float(get_value(cfg, "physical.dq", 0.605))
        self.k_yaw = float(get_value(cfg, "model.k_yaw", 0.8))
        
        # 推力参数
        self.max_thrust = float(get_value(cfg, "sim.max_thrust", 17.75))
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.dt
        
        # 初始化NMPC控制器
        self.controller = NMPC_Controller(config_path)
        
        # 初始化轨迹生成器
        self.trajectory_gen = TrajectoryGenerator()
        
        # 电机失能设置
        self.motor_failure_active = False
        self.motor_power_ratios = np.ones(8)  # 8个电机的功率比例 [0,1]，1表示正常
        self.failure_start_time = None
        
        # 数据记录
        self.data_log = []
        
        # 电机名称映射（8个独立电机 - 同轴反转配置）
        self.motor_names = [
            'motor0_front_upper_cw',    # 0: Front上,CW
            'motor1_left_upper_ccw',    # 1: Left上,CCW
            'motor2_rear_upper_cw',     # 2: Rear上,CW
            'motor3_right_upper_ccw',   # 3: Right上,CCW
            'motor4_front_lower_ccw',   # 4: Front下,CCW
            'motor5_left_lower_cw',     # 5: Left下,CW
            'motor6_rear_lower_ccw',    # 6: Rear下,CCW
            'motor7_right_lower_cw'     # 7: Right下,CW
        ]
        
        print(f"✓ 电机失能仿真器初始化完成")
        print(f"  时间步长: {self.dt}s")
        print(f"  质量: {self.mass}kg")
    
    def get_state(self):
        """获取无人机当前状态"""
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()  # [qw, qx, qy, qz]
        vel = self.data.qvel[0:3].copy()
        omega = self.data.qvel[3:6].copy()
        
        state = np.concatenate([pos, quat, vel, omega])
        return state
    
    def calc_motor_force(self, krpm):
        """根据电机转速计算推力"""
        return self.Ct * krpm**2
    
    def calc_motor_input_normalized(self, krpm):
        """将电机转速(krpm)转换为归一化控制输入[0,1]"""
        max_speed = self.controller.max_speed
        krpm = np.clip(krpm, 0.0, max_speed)
        force = self.calc_motor_force(krpm)
        normalized = force / self.max_thrust
        return np.clip(normalized, 0.0, 1.0)
    
    def calc_motor_input(self, nmpc_output):
        """计算电机输入（包含失能逻辑）
        
        Args:
            nmpc_output: NMPC输出 [w_front, w_left, w_rear, w_right, yaw_bias]
        
        Returns:
            motor_speeds_normalized: 8个电机归一化输入 [0,1]
        """
        w_front, w_left, w_rear, w_right, yaw_bias = nmpc_output
        
        # 基准混控（8电机同轴反转配置）
        # ⚠️ 必须使用乘法混控与NMPC模型保持一致！
        # Front arm: motor0(上,CW) + motor4(下,CCW)
        # Left arm:  motor1(上,CCW) + motor5(下,CW)
        # Rear arm:  motor2(上,CW) + motor6(下,CCW)
        # Right arm: motor3(上,CCW) + motor7(下,CW)
        motor_speeds = np.array([
            w_front * (1 - self.k_yaw * yaw_bias),  # motor0: Front上,CW
            w_left * (1 + self.k_yaw * yaw_bias),   # motor1: Left上,CCW
            w_rear * (1 - self.k_yaw * yaw_bias),   # motor2: Rear上,CW
            w_right * (1 + self.k_yaw * yaw_bias),  # motor3: Right上,CCW
            w_front * (1 + self.k_yaw * yaw_bias),  # motor4: Front下,CCW
            w_left * (1 - self.k_yaw * yaw_bias),   # motor5: Left下,CW
            w_rear * (1 + self.k_yaw * yaw_bias),   # motor6: Rear下,CCW
            w_right * (1 - self.k_yaw * yaw_bias)   # motor7: Right下,CW
        ])
        
        # 限幅
        motor_speeds = np.clip(motor_speeds, 0.0, self.controller.max_speed)
        
        # 应用电机功率降级
        motor_speeds = motor_speeds * self.motor_power_ratios
        
        # 转换为归一化输入[0,1]
        motor_inputs = np.array([self.calc_motor_input_normalized(s) for s in motor_speeds])
        
        return motor_inputs
    
    def set_motor_failure(self, motor_degradation_list):
        """设置电机功率降级
        
        Args:
            motor_degradation_list: 电机降级列表，格式: [[motor_idx, power_ratio], ...]
                                   例如: [[0, 0.5], [1, 0]] 表示motor0降到50%，motor1完全失效
        """
        self.motor_failure_active = True
        self.motor_power_ratios = np.ones(8)  # 重置为全部正常
        
        for motor_idx, power_ratio in motor_degradation_list:
            if motor_idx not in range(8):
                print(f"⚠️ 无效的电机索引: {motor_idx}，应为0-7")
                continue
            if not (0.0 <= power_ratio <= 1.0):
                print(f"⚠️ 无效的功率比例: {power_ratio}，应为0.0-1.0")
                continue
            
            self.motor_power_ratios[motor_idx] = power_ratio
        
        self.failure_start_time = self.data.time
        
        print(f"\n🔴 电机功率降级触发!")
        for motor_idx, power_ratio in motor_degradation_list:
            if motor_idx in range(8):
                motor_name = self.motor_names[motor_idx]
                status = "完全失效" if power_ratio == 0 else f"降到{power_ratio*100:.0f}%"
                print(f"  {motor_name}: {status}")
        print(f"  触发时间: {self.failure_start_time:.2f}s")
    
    def clear_motor_failure(self):
        """清除电机功率降级"""
        self.motor_failure_active = False
        self.motor_power_ratios = np.ones(8)
        self.failure_start_time = None
        print("\n✓ 电机功率降级已清除")
    
    def run_simulation(self, duration, trajectory_mode='hover', hover_position=None,
                      failure_time=None, motor_degradation=None,
                      render=True):
        """运行仿真
        
        Args:
            duration: 仿真时长(s)
            trajectory_mode: 轨迹模式 ('hover', 'circle', 'square', 'climb')
            hover_position: 悬停位置 [x, y, z]（仅在hover模式下使用）
            failure_time: 触发失能的时间点(s)，None表示不触发
            motor_degradation: 电机降级列表 [[motor_idx, power_ratio], ...]
            render: 是否可视化
        """
        if hover_position is None:
            hover_position = [0.0, 0.0, 1.0]
        
        # 设置轨迹模式
        if trajectory_mode == 'climb':
            self.trajectory_gen.set_climb_params(start_height=0.0, target_height=2.5, speed=1.8)
        self.trajectory_gen.set_mode(trajectory_mode)
        
        print(f"\n{'='*60}")
        print(f"开始电机失能仿真")
        print(f"  仿真时长: {duration}s")
        print(f"  轨迹模式: {trajectory_mode}")
        if trajectory_mode == 'hover':
            print(f"  悬停位置: {hover_position}")
        elif trajectory_mode == 'circle':
            print(f"  圆形轨迹: 半径={self.trajectory_gen.circle_radius}m, 周期={self.trajectory_gen.circle_period}s")
        elif trajectory_mode == 'square':
            print(f"  方形轨迹: 边长={self.trajectory_gen.square_size}m, 周期={self.trajectory_gen.square_period}s")
        elif trajectory_mode == 'climb':
            print(f"  爬升轨迹: 0m → 2.5m")
        
        if failure_time is not None and motor_degradation is not None:
            print(f"  失能设置: t={failure_time}s")
            for motor_idx, ratio in motor_degradation:
                if motor_idx in range(8):
                    status = "完全失效" if ratio == 0 else f"降到{ratio*100:.0f}%"
                    print(f"    {self.motor_names[motor_idx]}: {status}")
        else:
            print(f"  失能设置: 无（正常飞行）")
        print(f"{'='*60}\n")
        
        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)
        self.data_log = []
        self.motor_failure_active = False
        self.motor_power_ratios = np.ones(8)
        
        # 设置初始位置
        if trajectory_mode == 'climb':
            self.data.qpos[0:3] = [0.0, 0.0, 0.0]  # 从地面起飞
        else:
            self.data.qpos[0:3] = hover_position
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 水平姿态
        
        # 创建viewer
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            viewer.cam.distance = 8.0
            viewer.cam.elevation = -30
        else:
            viewer = None
        
        step_count = 0
        last_print_time = 0.0
        
        try:
            while self.data.time < duration:
                # 检查是否触发电机失能
                if (failure_time is not None and motor_degradation is not None and 
                    not self.motor_failure_active and 
                    self.data.time >= failure_time):
                    self.set_motor_failure(motor_degradation)
                
                # 获取当前状态
                current_state = self.get_state()
                
                # 从轨迹生成器获取目标
                if trajectory_mode == 'hover':
                    goal_pos = np.array(hover_position)
                    goal_vel = np.zeros(3)
                    goal_quat = np.array([1.0, 0.0, 0.0, 0.0])
                else:
                    goal_pos, goal_vel, goal_yaw, goal_yaw_rate = self.trajectory_gen.get_reference_state(self.data.time)
                    goal_quat = np.array([np.cos(goal_yaw/2), 0, 0, np.sin(goal_yaw/2)])
                
                # NMPC控制
                _, control = self.controller.nmpc_position_control(
                    current_state, goal_pos, goal_vel, goal_quat
                )
                
                # 计算电机输入（含失能逻辑）
                motor_inputs = self.calc_motor_input(control)
                
                # 应用控制
                # MuJoCo模型有18个actuator: 8个motor joints + 2个arm servos + 8个propellers
                # 我们只控制推进器 (索引10-17)，输入为归一化值[0,1]
                self.data.ctrl[10:18] = motor_inputs
                # 保持舵机固定（默认值0）
                self.data.ctrl[8:10] = 0.0
                
                # 步进仿真
                mujoco.mj_step(self.model, self.data)
                step_count += 1
                
                # 记录数据
                pos = current_state[0:3]
                quat = current_state[3:7]
                vel = current_state[7:10]
                omega = current_state[10:13]
                
                # 计算欧拉角（用于可视化）
                rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # [qx,qy,qz,qw]
                euler = rot.as_euler('xyz', degrees=True)
                
                log_entry = {
                    'time': self.data.time,
                    'pos': pos.copy(),
                    'quat': quat.copy(),
                    'euler': euler.copy(),
                    'vel': vel.copy(),
                    'omega': omega.copy(),
                    'control': control.copy(),
                    'motor_inputs': motor_inputs.copy(),  # 归一化输入[0,1]
                    'motor_failure': self.motor_failure_active,
                    'motor_power_ratios': self.motor_power_ratios.copy()
                }
                self.data_log.append(log_entry)
                
                # 更新显示
                if render and viewer is not None:
                    viewer.sync()
                
                # 定期打印状态
                if self.data.time - last_print_time >= 1.0:
                    status = "失能" if self.motor_failure_active else "正常"
                    print(f"t={self.data.time:6.2f}s | 位置: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                          f"欧拉角: [{euler[0]:6.1f}°, {euler[1]:6.1f}°, {euler[2]:6.1f}°] | "
                          f"状态: {status}")
                    last_print_time = self.data.time
        
        except KeyboardInterrupt:
            print("\n⚠️ 仿真被用户中断")
        
        finally:
            if render and viewer is not None:
                viewer.close()
        
        print(f"\n✓ 仿真完成: {step_count}步, {self.data.time:.2f}s")
        print(f"  记录数据点: {len(self.data_log)}")
    
    def save_data(self, filename='motor_failure_data.npz'):
        """保存仿真数据"""
        if not self.data_log:
            print("⚠️ 没有数据可保存")
            return
        
        # 提取数据
        time = np.array([d['time'] for d in self.data_log])
        pos = np.array([d['pos'] for d in self.data_log])
        quat = np.array([d['quat'] for d in self.data_log])
        euler = np.array([d['euler'] for d in self.data_log])
        vel = np.array([d['vel'] for d in self.data_log])
        omega = np.array([d['omega'] for d in self.data_log])
        control = np.array([d['control'] for d in self.data_log])
        motor_inputs = np.array([d['motor_inputs'] for d in self.data_log])
        motor_failure = np.array([d['motor_failure'] for d in self.data_log])
        motor_power_ratios = np.array([d['motor_power_ratios'] for d in self.data_log])
        
        # 保存
        np.savez(filename,
                 time=time,
                 pos=pos,
                 quat=quat,
                 euler=euler,
                 vel=vel,
                 omega=omega,
                 control=control,
                 motor_inputs=motor_inputs,  # 归一化输入[0,1]
                 motor_failure=motor_failure,
                 motor_power_ratios=motor_power_ratios)
        
        print(f"✓ 数据已保存到: {filename}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='八旋翼电机失能仿真')
    parser.add_argument('--mode', default='hover', choices=['hover','circle','square','climb'], help='轨迹模式')
    parser.add_argument('--duration', type=float, default=40.0, help='仿真时长(秒)')
    parser.add_argument('--time', type=float, default=None, help='失能触发时间(秒)')
    parser.add_argument('--degradation', type=str, default=None, help='电机降级 "[[0,0.5],[1,0]]"')
    parser.add_argument('--no-render', action='store_true', help='禁用可视化')
    args = parser.parse_args()
    
    motor_degradation = None
    if args.degradation:
        try:
            import ast
            motor_degradation = ast.literal_eval(args.degradation)
        except:
            pass
    
    scene_path = join(dirname(abspath(__file__)), "robots/haique.xml")
    sim = MotorFailureSimulator(scene_path)
    
    sim.run_simulation(
        duration=args.duration,
        trajectory_mode=args.mode,
        hover_position=[0.0, 0.0, 1.0],
        failure_time=args.time,
        motor_degradation=motor_degradation,
        render=not args.no_render
    )
    
    sim.save_data('motor_failure_data.npz')
    print("\n" + "="*60)
    print("仿真完成！运行 visualize_motor_failure.py 查看结果")
    print("="*60)


if __name__ == '__main__':
    main()
