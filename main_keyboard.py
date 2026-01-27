#!/usr/bin/env python3
"""
键盘控制主程序 - WSL环境下使用键盘代替手柄
"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
from os.path import abspath, dirname, join
from control.keyboard_controller import KeyboardController
from controller_state_machine import ControllerStateMachine
from control.nmpc_controller import NMPC_Controller as AerialController
from control.nmpc_controller_underwater import NMPC_Controller as UnderwaterController
from model.config_loader import get_mode_config, get_value
from visualization.keyboard_control_plotter import KeyboardControlLogger, KeyboardControlPlotter

CONFIG_PATH = join(dirname(abspath(__file__)), "config.yaml")

def yaw_to_quaternion(yaw, reference_yaw=None):
    """
    将yaw角度转换为四元数。
    如果提供reference_yaw，会确保yaw与reference的差值在±π范围内。
    
    Args:
        yaw: 偏航角（弧度），可以是任意值
        reference_yaw: 参考yaw角度（弧度），用于避免跳变
        
    Returns:
        四元数 [w, x, y, z]，表示绕z轴的旋转
    """
    # if reference_yaw is not None:
    #     # 计算差值并归一化到[-π, π]
    #     diff = yaw - reference_yaw
    #     diff = np.arctan2(np.sin(diff), np.cos(diff))
    #     # 调整yaw使其接近reference
    #     yaw = reference_yaw + diff
    
    # 不再归一化yaw，保持连续性以支持连续旋转
    # 四元数会自动处理周期性（q 和 -q 表示同一姿态）
    half_yaw = yaw / 2.0
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])

def main(enable_plot=True, plot_interval=100):
    # 加载配置
    cfg_aerial = get_mode_config("aerial", path=CONFIG_PATH)
    cfg_underwater = get_mode_config("underwater", path=CONFIG_PATH)
    
    # 物理参数（aerial模式）
    Ct_aerial = float(get_value(cfg_aerial, "physical.Ct", 0.1757))
    max_thrust_aerial = float(get_value(cfg_aerial, "sim.max_thrust", 17.75))
    max_speed_aerial = float(get_value(cfg_aerial, "nmpc.max_speed", 22.0))
    k_yaw = float(get_value(cfg_aerial, "model.k_yaw", 0.8))
    
    # 物理参数（underwater模式）
    Ct_underwater = float(get_value(cfg_underwater, "physical.Ct", 0.0267))
    max_thrust_underwater = float(get_value(cfg_underwater, "sim.max_thrust", 24.0))
    max_speed_underwater = float(get_value(cfg_underwater, "nmpc.max_speed", 30.0))
    
    # 舵机参数
    left_servo_offset = float(get_value(cfg_underwater, "servo.left_offset", -0.1))
    right_servo_offset = float(get_value(cfg_underwater, "servo.right_offset", 0.0))
    servo_lpf_tau = float(get_value(cfg_underwater, "sim.servo_lpf_tau", 0.06))
    servo_rate_limit = float(get_value(cfg_underwater, "sim.servo_rate_limit", 3.0))
    
    print("初始化控制器...")
    
    # 初始化双控制器
    aerial_ctrl = AerialController(config_path=CONFIG_PATH)
    underwater_ctrl = UnderwaterController(
        left_servo_offset=left_servo_offset,
        right_servo_offset=right_servo_offset,
        config_path=CONFIG_PATH
    )
    
    # 初始化状态机
    state_machine = ControllerStateMachine(
        vel_threshold=0.05,
        hysteresis_time=0.3,
        dt=0.01
    )
    
    # 启动仿真
    print("加载Haique模型...")
    model = mujoco.MjModel.from_xml_path("./robots/haique.xml")
    data = mujoco.MjData(model)
    
    # 初始化键盘控制
    keyboard = KeyboardController(
        max_vel_xy=0.8,
        max_vel_z=0.8,
        max_yaw_rate=0.5,
        accel_rate=2.0
    )
    
    # 启动键盘控制
    if not keyboard.start():
        print("❌ 键盘初始化失败")
        return
    
    print("\n" + "="*60)
    print("✅ 键盘控制主程序已启动")
    print("="*60)
    print("按H查看键盘控制帮助")
    print("正在启动MuJoCo可视化...")
    print("="*60 + "\n")
    
    # 目标位置累积器（用于速度积分）
    target_pos = np.array([0.0, 0.0, 0.5])
    target_yaw = 0.0  # 目标偏航角
    yaw_continuous = 0.0  # 连续的实际yaw角（unwrap处理）
    last_yaw_raw = 0.0  # 上一次的原始yaw角
    
    # 初始化数据记录和绘图
    logger = KeyboardControlLogger('log/keyboard_control_log.csv')
    plotter = KeyboardControlPlotter(buffer_size=500) if enable_plot else None
    
    # 舵机滤波状态
    servo_cmd_filt = np.array([
        (np.pi / 2) + left_servo_offset,
        (np.pi / 2) + right_servo_offset,
    ], dtype=float)
    
    last_quat_main = np.array([1.0, 0.0, 0.0, 0.0])
    
    # 电机输入计算函数
    def calc_motor_input_aerial(krpm):
        krpm = np.clip(krpm, 0.0, max_speed_aerial)
        force = Ct_aerial * krpm**2
        return np.clip(force / max_thrust_aerial, 0.0, 1.0)
    
    def calc_motor_input_underwater(krpm, idx):
        krpm = np.clip(krpm, -max_speed_underwater, max_speed_underwater)
        force = Ct_underwater * krpm * np.abs(krpm)
        input_val = force / max_thrust_underwater
        
        if idx % 2 == 0:  # 偶数索引：单向
            return max(input_val, 0.0)
        else:  # 奇数索引：双向
            return np.clip(input_val, -1.0, 1.0)
    
    # 启动MuJoCo被动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -20
        
        print("✅ MuJoCo可视化已启动")
        print("⌨️  开始键盘控制（焦点需在pygame窗口）\n")
        
        try:
            step = 0
            dt = model.opt.timestep  # 定义dt在循环外
            while viewer.is_running():
                step_start = time.time()
                
                # 获取键盘输入
                vel_cmd, is_active = keyboard.get_velocity_cmd()
                
                # 检查紧急停止
                if keyboard.is_emergency_stopped():
                    vel_cmd = np.zeros(4)
                
                # 当前状态
                pos = data.qpos[:3].copy()
                quat = data.qpos[3:7].copy()
                vel = data.qvel[:3].copy()
                omega = data.qvel[3:6].copy()
                
                # 四元数翻转修正
                if np.dot(quat, last_quat_main) < 0:
                    quat = -quat
                last_quat_main = quat.copy()
                
                # 计算欧拉角 (roll, pitch, yaw)
                qw, qx, qy, qz = quat
                roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
                pitch = np.arcsin(2*(qw*qy - qz*qx))
                yaw_raw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
                
                # 处理yaw角度连续化（unwrap）：检测-π到π的跳变
                yaw_diff = yaw_raw - last_yaw_raw
                if yaw_diff > np.pi:
                    yaw_continuous -= 2 * np.pi
                elif yaw_diff < -np.pi:
                    yaw_continuous += 2 * np.pi
                yaw_continuous += yaw_diff
                last_yaw_raw = yaw_raw
                
                # 用于显示的欧拉角（度）
                euler_deg = np.degrees([roll, pitch, yaw_raw])
                target_yaw_deg = np.degrees(target_yaw)
                
                current_state = np.concatenate([pos, quat, vel, omega])
                
                
                # 将机体坐标系速度转换为世界坐标系速度
                dt = model.opt.timestep
                body_vx = vel_cmd[0]  # 机体前进速度
                body_vy = vel_cmd[1]  # 机体侧向速度（当前为0）
                
                # 转换到世界坐标系
                world_vx = body_vx * np.cos(target_yaw) - body_vy * np.sin(target_yaw)
                world_vy = body_vx * np.sin(target_yaw) + body_vy * np.cos(target_yaw)
                world_vz = vel_cmd[2]  # z方向速度不变
                
                # 更新目标位置（速度积分）
                target_pos[0] += world_vx * dt
                target_pos[1] += world_vy * dt
                target_pos[2] += world_vz * dt
                target_yaw += vel_cmd[3] * dt  # 偏航角速率积分（完全连续，无归一化）
                
                # 目标速度也使用世界坐标系
                target_vel = np.array([world_vx, world_vy, world_vz])
                
                # 状态机选择控制器
                mode, state_changed = state_machine.update(vel_cmd[:4])
                
                # 模式切换处理：切换到hover时，继承当前yaw角作为目标
                if state_changed and mode == ControllerStateMachine.STATE_HOVER:
                    target_yaw = yaw_continuous  # 使用当前连续yaw作为hover目标
                    print(f"    → 切换到hover模式，锁定目标yaw = {np.degrees(target_yaw):.1f}°")
                
                # 记录数据和更新绘图
                logger.log_data(
                    time=step * dt,
                    mode=mode,
                    vel_cmd=vel_cmd,
                    pos=pos,
                    euler=[roll, pitch, yaw_continuous],  # 使用连续的yaw角
                    vel=vel,
                    target_yaw=target_yaw
                )
                if plotter:
                    plotter.update_data(
                        time=step * dt,
                        vel_cmd=vel_cmd,
                        pos=pos,
                        vel=vel,
                        yaw=yaw_continuous,  # 使用连续的yaw角
                        target_yaw=target_yaw
                    )
                
                # ========== 模式分支处理 ==========
                if mode == ControllerStateMachine.STATE_HOVER:
                    # 悬停模式：aerial控制器
                    # 将target_yaw转换为四元数（与实际yaw保持在同一周期）
                    goal_quat = yaw_to_quaternion(target_yaw, reference_yaw=yaw_continuous)
                    
                    _dt, u = aerial_ctrl.nmpc_position_control(
                        current_state=current_state,
                        goal_pos=target_pos,
                        goal_vel=target_vel,
                        goal_quat=goal_quat,
                        disturbance=None
                    )
                    
                    # u = [w_front, w_left, w_rear, w_right, yaw_bias]
                    yaw_bias = float(u[4])
                    motor_speeds = np.array([
                        u[0] * (1 - k_yaw * yaw_bias),  # motor0: Front上,CW
                        u[1] * (1 + k_yaw * yaw_bias),  # motor1: Left上,CCW
                        u[2] * (1 - k_yaw * yaw_bias),  # motor2: Rear上,CW
                        u[3] * (1 + k_yaw * yaw_bias),  # motor3: Right上,CCW
                        u[0] * (1 + k_yaw * yaw_bias),  # motor4: Front下,CCW
                        u[1] * (1 - k_yaw * yaw_bias),  # motor5: Left下,CW
                        u[2] * (1 + k_yaw * yaw_bias),  # motor6: Rear下,CCW
                        u[3] * (1 - k_yaw * yaw_bias),  # motor7: Right下,CW
                    ])
                    motor_speeds = np.clip(motor_speeds, 0.0, max_speed_aerial)
                    
                    # 应用电机控制
                    for i in range(8):
                        data.actuator(f'prop_motor{i}').ctrl[0] = calc_motor_input_aerial(motor_speeds[i])
                    
                    # 舵机固定为90度
                    data.actuator('left_servo').ctrl[0] = left_servo_offset
                    data.actuator('right_servo').ctrl[0] = right_servo_offset
                
                else:
                    # 运动模式：underwater控制器
                    # 将target_yaw转换为四元数（与实际yaw保持在同一周期）
                    goal_quat = yaw_to_quaternion(target_yaw, reference_yaw=yaw_continuous)
                    
                    _dt, u = underwater_ctrl.nmpc_position_control(
                        current_state=current_state,
                        goal_pos=target_pos,
                        goal_vel=target_vel,
                        goal_quat=goal_quat,
                        disturbance=None
                    )
                    
                    # u = [motor0-7, servo_left, servo_right]
                    motor_speeds = u[:8]
                    
                    # 应用电机控制
                    for i in range(8):
                        data.actuator(f'prop_motor{i}').ctrl[0] = calc_motor_input_underwater(motor_speeds[i], i)
                    
                    # 舵机指令整形（低通+限速）
                    servo_des = np.array([
                        u[8] + left_servo_offset,
                        u[9] + right_servo_offset,
                    ])
                    
                    # 一阶低通
                    alpha = dt / (servo_lpf_tau + dt)
                    servo_lpf = servo_cmd_filt + alpha * (servo_des - servo_cmd_filt)
                    
                    # 速率限制
                    max_step = servo_rate_limit * dt
                    step_diff = np.clip(servo_lpf - servo_cmd_filt, -max_step, max_step)
                    servo_cmd_filt = servo_cmd_filt + step_diff
                    servo_cmd_filt = np.clip(servo_cmd_filt, 0.0, np.pi)
                    
                    data.actuator('left_servo').ctrl[0] = float(servo_cmd_filt[0])
                    data.actuator('right_servo').ctrl[0] = float(servo_cmd_filt[1])
                
                # 仿真步进
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 打印状态（每100步）
                if step % 100 == 0:
                    print(f"t={step*dt:.1f}s | 模式:{mode:10s} | "
                          f"pos:[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}] | "
                          f"目标:[{target_pos[0]:.2f},{target_pos[1]:.2f},{target_pos[2]:.2f}] | "
                          f"欧拉角:[R:{euler_deg[0]:.1f}° P:{euler_deg[1]:.1f}° Y:{euler_deg[2]:.1f}°] | "
                          f"目标Yaw:{target_yaw_deg:.1f}° | "
                          f"速度:[{vel_cmd[0]:.1f},{vel_cmd[1]:.1f},{vel_cmd[2]:.1f}]")
                
                # 实时绘图更新（降低更新频率以提高帧率）
                if plotter and step % plot_interval == 0:
                    plotter.update_plot()
                
                if state_changed and mode != ControllerStateMachine.STATE_HOVER:
                    print(f"    → 切换到 {mode} 模式")
                
                step += 1
                
                # 控制仿真频率
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        except KeyboardInterrupt:
            print("\n\n用户中断")
        
        finally:
            keyboard.stop()
            logger.close()
            if plotter:
                plotter.close()
            print(f"\n✓ 日志已保存到: log/keyboard_control_log.csv")
            print("✓ 程序已退出")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keyboard control with optional plotting')
    parser.add_argument('--no-plot', action='store_true', help='Disable real-time plotting')
    parser.add_argument('--plot-interval', type=int, default=100, 
                        help='Plot update interval in steps (default: 100, lower = more frequent but slower)')
    args = parser.parse_args()
    
    main(enable_plot=not args.no_plot, plot_interval=args.plot_interval)
