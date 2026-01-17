# 20250220 Wakkk
# Quadrotor SE3 Control Demo
import mujoco 
import mujoco.viewer as viewer 
import numpy as np
from os.path import abspath, dirname, join
from nmpc_controller_underwater import NMPC_Controller
from trajectory_generator import TrajectoryGenerator
from eso_observer import ESO_Observer
from disturbance_generator import DisturbanceGenerator, DisturbanceScenarios

from config_loader import get_mode_config, get_value


CONFIG_PATH = join(dirname(abspath(__file__)), "config.yaml")
CFG = get_mode_config("underwater", path=CONFIG_PATH)

left_servo_offset = float(get_value(CFG, "servo.left_offset", -0.1))
right_servo_offset = float(get_value(CFG, "servo.right_offset", 0.0))

# 新建NMPC控制器（将舵机安装偏置传入控制器，使“pi/2”对应物理舵机角）
controller = NMPC_Controller(
    left_servo_offset=left_servo_offset,
    right_servo_offset=right_servo_offset,
    config_path=CONFIG_PATH,
)

# 新建轨迹生成器
trajectory_gen = TrajectoryGenerator()

gravity = float(get_value(CFG, "physical.g", 9.8066))
mass = float(get_value(CFG, "physical.mass", 4.672))
Ct = float(get_value(CFG, "physical.Ct", 0.0267))
Cd = float(get_value(CFG, "physical.Cd", 0.00111))

dq = float(get_value(CFG, "physical.dq", 0.605))
arm_length = dq / 2.0
max_thrust = float(get_value(CFG, "sim.max_thrust", 24.0))
max_torque = float(get_value(CFG, "sim.max_torque", 0.02))

# 控制周期（NMPC/ESO 的离散时间假设）
control_dt = float(get_value(CFG, "sim.control_dt", 0.01))

# 舵机指令整形：NMPC 的输出会有高频微抖，直接给 MuJoCo position actuator 会激发振荡。
# 这里做最小处理：一阶低通 + 速率限制（单位：rad, rad/s）。
servo_lpf_tau = float(get_value(CFG, "sim.servo_lpf_tau", 0.06))
servo_rate_limit = float(get_value(CFG, "sim.servo_rate_limit", 3.0))  # rad/s

# 新建ESO扰动观测器（与 control_dt 保持一致）
eso = ESO_Observer(
    mode="underwater",
    dt=control_dt,
    left_servo_offset=left_servo_offset,
    right_servo_offset=right_servo_offset,
    config_path=CONFIG_PATH,
)

# 根据电机转速计算电机推力
def calc_motor_force(krpm):
    global Ct
    return Ct * krpm * np.abs(krpm)

# 根据电机转速计算电机归一化输入
# 偶数索引电机（0,2,4,6）保持单向推力，奇数索引电机允许反向（-1~1）
def calc_motor_input(krpm, idx):
    max_speed = float(get_value(CFG, "nmpc.max_speed", 30.0))
    krpm = np.clip(krpm, -max_speed, max_speed)

    # 计算推力（带符号），再按最大正向推力归一化
    _force = calc_motor_force(krpm)
    _input = _force / max_thrust  # 期望范围约 [-1, 1]

    if idx % 2 == 0:
        # 单向：只保留推力方向
        _input = max(_input, 0.0)
    else:
        # 允许反向：夹到 [-1, 1]
        _input = np.clip(_input, -1.0, 1.0)

    return _input

# 全局变量用于键盘输入
last_key_time = 0.0
key_press_cooldown = 0.5  # 按键冷却时间(秒)

# 加载模型回调函数
def load_callback(m=None, d=None):
    mujoco.set_mjcb_control(None)
    m = mujoco.MjModel.from_xml_path('./crazyfile/scene.xml')
    d = mujoco.MjData(m)
    if m is not None:
        mujoco.set_mjcb_control(lambda m, d: control_callback(m, d))  # 设置控制回调函数
    
    # 设置默认摄像头为tracking模式
    # 查找track_cam的ID
    track_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'track_cam')
    if track_cam_id >= 0:
        print(f"✓ 设置默认摄像头: track_cam (ID={track_cam_id})")
    
    return m, d

# 根据四元数计算旋转矩阵
def rotation_matrix(q0, q1, q2, q3):
    _row0 = np.array([1-2*(q2**2)-2*(q3**2), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)])
    _row1 = np.array([2*(q1*q2+q0*q3), 1-2*(q1**2)-2*(q3**2), 2*(q2*q3-q0*q1)])
    _row2 = np.array([2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1**2)-2*(q2**2)])
    return np.vstack((_row0, _row1, _row2))

log_count = 0
eso_enable = True  # 默认启用ESO（可通过命令行参数修改）

# 扰动生成器（可选，用于测试控制器鲁棒性）
disturbance_gen = None
disturbance_enable = False

# NMPC/ESO 以 control_dt 更新，其余仿真步保持上一控制量
last_nmpc_time = -1.0
held_control = np.zeros(10)
last_control_for_eso = np.zeros(10)

last_quat_main = np.array([1.0, 0.0, 0.0, 0.0])

# 记录舵机“实际下发”的滤波值（MuJoCo ctrl 是 joint target，单位 rad）
servo_cmd_filt = np.array([
    (np.pi / 2) + left_servo_offset,
    (np.pi / 2) + right_servo_offset,
], dtype=float)

def control_callback(m, d):
    global log_count, gravity, mass, controller, trajectory_gen, eso, eso_enable
    global last_nmpc_time, held_control, last_control_for_eso, last_quat_main, servo_cmd_filt

    pos = d.qpos[:3]        # [x, y, z]
    quat = d.qpos[3:7]      # [qw, qx, qy, qz]
    vel = d.qvel[:3]        # [vx, vy, vz]
    omega = d.qvel[3:6]     # [wx, wy, wz]
    state_obs = d.qvel[:6]

    if np.dot(quat, last_quat_main) < 0:
        quat = -quat
    last_quat_main = quat.copy()

    current_state = np.concatenate([pos, quat, vel, omega])

    # 构建当前状态 for Controller (matches nmpc_controller expectations)
    # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    current_state = np.concatenate([pos, quat, vel, omega])
    
    # 控制更新节拍：MuJoCo 可能是 500Hz（timestep=0.002），这里按 control_dt 运行 NMPC/ESO。
    do_update = (last_nmpc_time < 0.0) or ((d.time - last_nmpc_time) >= (control_dt - 1e-9))

    if do_update:
        last_nmpc_time = float(d.time)

        # 从轨迹生成器获取目标位置 + 速度前馈 + yaw + yaw_rate
        goal_position, goal_velocity, goal_yaw, goal_yaw_rate = trajectory_gen.get_reference_state(d.time)
        
        # 将yaw转换为四元数
        def yaw_to_quat(yaw):
            half_yaw = yaw / 2.0
            return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
        
        goal_quat = yaw_to_quat(goal_yaw)

        # 获取实际施加的扰动（用于记录）
        global disturbance_gen, disturbance_enable
        if disturbance_enable and disturbance_gen is not None:
            actual_dist = disturbance_gen.get_disturbance(d.time)
            actual_disturbance = {'force': actual_dist['force'].copy(), 'torque': actual_dist['torque'].copy()}
        else:
            actual_disturbance = None

        # 获取ESO扰动估计（可选用于前馈补偿）
        if eso_enable:
            dist_f, dist_m = eso.update(state_obs, last_control_for_eso, quat)
            eso_disturbance = {'force': dist_f, 'torque': dist_m}
        else:
            eso_disturbance = None

        _solve_dt, new_control = controller.nmpc_position_control(
            current_state, goal_position, eso_disturbance, 
            goal_vel=goal_velocity, goal_quat=goal_quat, goal_yaw_rate=goal_yaw_rate,
            actual_disturbance=actual_disturbance
        )
        held_control = new_control.copy()
        last_control_for_eso = held_control.copy()

    _control = held_control
    
    # 计算实际的8个电机转速
    motor_speeds = np.array([
        _control[0],  # motor0: Front上,CW
        _control[1],  # motor1: Left上,CCW
        _control[2],  # motor2: Rear上,CW
        _control[3],  # motor3: Right上,CCW
        _control[4],  # motor4: Front下,CCW
        _control[5],  # motor5: Left下,CW
        _control[6],  # motor6: Rear下,CCW
        _control[7],  # motor7: Right下,CW
    ])

    # 测试用例：全部电机关闭
    # motor_speeds = np.zeros_like(motor_speeds)
    
    # === 应用外部扰动（如果启用）===
    if disturbance_enable and disturbance_gen is not None:
        dist = disturbance_gen.get_disturbance(d.time)
        # 将扰动力和力矩直接施加到无人机
        # force: [fx, fy, fz] in world frame (N)
        # torque: [mx, my, mz] in body frame (Nm)
        d.xfrc_applied[1, :3] = dist['force']   # body 1 是无人机主体
        d.xfrc_applied[1, 3:] = dist['torque']
    
    # 应用电机控制
    for i in range(8):
        d.actuator(f'prop_motor{i}').ctrl[0] = calc_motor_input(motor_speeds[i], i)

    # ---- 舵机指令整形：低通 + 限速 + 夹到 joint range ----
    servo_des = np.array([
        _control[8] + left_servo_offset,
        _control[9] + right_servo_offset,
    ], dtype=float)

    # 一阶低通
    sim_dt = float(m.opt.timestep)
    alpha = sim_dt / (servo_lpf_tau + sim_dt)
    servo_lpf = servo_cmd_filt + alpha * (servo_des - servo_cmd_filt)

    # 速率限制
    max_step = servo_rate_limit * sim_dt
    step = np.clip(servo_lpf - servo_cmd_filt, -max_step, max_step)
    servo_cmd_filt = servo_cmd_filt + step

    # 夹紧到 joint 的物理范围（haique.xml 里已是 [0, pi]）
    servo_cmd_filt = np.clip(servo_cmd_filt, 0.0, float(np.pi))

    d.actuator('left_servo').ctrl[0] = float(servo_cmd_filt[0])
    d.actuator('right_servo').ctrl[0] = float(servo_cmd_filt[1])

    # 测试用例：舵机固定位置
    # d.actuator('left_servo').ctrl[0] = np.pi/2 + left_servo_offset   # ~90 degrees
    # d.actuator('right_servo').ctrl[0] = np.pi/2 + right_servo_offset
    
    log_count += 1
    if log_count >= 50:
        log_count = 0
        # 输出扰动估计（仅在ESO启用时）
        if eso_enable:
            dist = eso_disturbance
            # print(f"扰动力: {dist['force']}, 扰动力矩: {dist['torque']}")

if __name__ == '__main__':
    import sys
    import argparse
    
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description='八旋翼NMPC控制器 - 轨迹跟踪模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
飞行模式:
  1 - 悬停模式 (Hover at 1.0m)
  2 - 画圆模式 (Circle)
  3 - 画方形模式 (Square)
  4 - 连续爬升 (Climb: 0m → 2.5m) ⭐推荐
  5 - 自动演示 (Auto Demo)

示例:
  python3 main.py 1              # 悬停模式，ESO启用（默认）
  python3 main.py 2 --no-eso     # 画圆模式，禁用ESO
  python3 main.py 4 --eso        # 爬升模式，显式启用ESO
        """
    )
    parser.add_argument('mode', type=str, nargs='?', default=None,
                        help='飞行模式 (1-6)')
    parser.add_argument('--eso', dest='eso_enable', action='store_true', 
                        default=True, help='启用ESO扰动观测器（默认启用）')
    parser.add_argument('--no-eso', dest='eso_enable', action='store_false',
                        help='禁用ESO扰动观测器')
    parser.add_argument('--disturbance', type=str, default=None,
                        choices=['mild', 'moderate', 'severe', 'random'],
                        help='启用外部扰动场景 (mild/moderate/severe/random)')
    
    args = parser.parse_args()
    
    # 设置全局ESO开关
    eso_enable = args.eso_enable
    
    # 设置扰动生成器（修改全局变量）
    if args.disturbance is not None:
        disturbance_enable = True
        disturbance_gen = DisturbanceGenerator(seed=42)
        
        if args.disturbance == 'mild':
            DisturbanceScenarios.mild_disturbance(disturbance_gen)
        elif args.disturbance == 'moderate':
            DisturbanceScenarios.moderate_disturbance(disturbance_gen)
        elif args.disturbance == 'severe':
            DisturbanceScenarios.severe_disturbance(disturbance_gen)
        elif args.disturbance == 'random':
            DisturbanceScenarios.random_disturbance(disturbance_gen, severity='moderate')
        
        disturbance_gen.print_summary()
    else:
        disturbance_enable = False
    
    print("="*80)
    print("🚁 八旋翼NMPC控制器 - 轨迹跟踪模式")
    print("="*80)
    print(f"\n【ESO扰动观测器】: {'✅ 启用' if eso_enable else '❌ 禁用'}")
    print(f"【外部扰动】: {'✅ 启用 (' + args.disturbance + ')' if disturbance_enable else '❌ 禁用'}")
    print("\n【选择飞行模式】")
    print("  1 - 悬停模式 (Hover at 1.0m)")
    print("  2 - 画圆模式 (Circle)")
    print("  3 - 画方形模式 (Square)")
    print("  4 - 连续爬升 (Climb: 0m → 2.5m, 从地面起飞) ⭐推荐")
    print("  5 - 自动演示 (Auto Demo)")
    print("  6 - 匀速前进 (Constant Forward)")
    
    # 如果有命令行参数，使用命令行参数，否则交互式输入
    if args.mode is not None:
        mode_choice = args.mode
    else:
        mode_choice = input("\n请选择模式 (1-6，默认1): ").strip() or "1"
    
    # 设置初始模式
    if mode_choice == "2":
        trajectory_gen.set_mode('circle')
        print(f"\n✓ 已选择: 画圆模式")
    elif mode_choice == "3":
        trajectory_gen.set_mode('square')
        print(f"\n✓ 已选择: 画方形模式")
    elif mode_choice == "4":
        # 连续爬升模式（从地面起飞）
        trajectory_gen.set_climb_params(start_height=0.0, target_height=2.5, speed=1.8)
        trajectory_gen.set_mode('climb')
        print(f"\n✓ 已选择: 连续爬升模式")
        print(f"  从地面(0m)平滑爬升到2.5m")
        print(f"  ⚠️  注意：无人机将从地面起飞！")
    elif mode_choice == "5":
        trajectory_gen.set_mode('hover')
        print(f"\n✓ 已选择: 自动演示模式")
        print("  → 程序将自动切换轨迹")
    elif mode_choice == "6":
        trajectory_gen.set_mode('forward')
        print(f"\n✓ 已选择: 匀速前进模式")
    else:
        trajectory_gen.set_mode('hover')
        print(f"\n✓ 已选择: 悬停模式")
    
    print("\n【轨迹参数】")
    print(f"  悬停位置: [0.0, 0.0, 1.0]")
    print(f"  圆形: 半径={trajectory_gen.circle_radius}m, 周期={trajectory_gen.circle_period}s")
    print(f"  方形: 边长={trajectory_gen.square_size}m, 周期={trajectory_gen.square_period}s")
    print("\n【摄像头控制】")
    print("  [ / ] - 切换摄像头视角")
    print("  鼠标右键拖动 - 调整视角")
    print("  滚轮 - 缩放")
    print("\n【运行时切换模式】")
    print("  修改代码第158行的 trajectory_gen.set_mode('模式')")
    print("  模式可选: 'hover', 'circle', 'square'")
    print("\n启动中...\n")
    print("="*80)
    
    # 自动演示模式的时间控制
    auto_demo_mode = (mode_choice == "5")
    
    # 修改control_callback以支持自动演示
    if auto_demo_mode:
        demo_start_time = [None]  # 使用列表以便在闭包中修改
        
        def auto_demo_control_callback(m, d):
            if demo_start_time[0] is None:
                demo_start_time[0] = d.time
            
            elapsed = d.time - demo_start_time[0]
            if elapsed < 10:
                # 前10秒悬停
                if trajectory_gen.mode != 'hover':
                    trajectory_gen.set_mode('hover')
            elif elapsed < 30:
                # 10-30秒画圆
                if trajectory_gen.mode != 'circle':
                    trajectory_gen.set_mode('circle')
            elif elapsed < 50:
                # 30-50秒画方形
                if trajectory_gen.mode != 'square':
                    trajectory_gen.set_mode('square')
            else:
                # 50秒后悬停
                if trajectory_gen.mode != 'hover':
                    trajectory_gen.set_mode('hover')
            
            return control_callback(m, d)
        
        # 替换控制回调
        mujoco.set_mjcb_control(lambda m, d: auto_demo_control_callback(m, d))
    
    try:
        viewer.launch(loader=load_callback)
    finally:
        # 保存数据
        print("\n正在保存飞行数据...")
        controller.save_data('./log/nmpc_data.csv')
        print("✓ NMPC数据已保存到 ./log/nmpc_data.csv")
        
        if eso_enable:
            eso.save_disturbance_log('./log/eso_disturbance_log.csv')
            print("✓ ESO扰动数据已保存到 ./log/eso_disturbance_log.csv")
        else:
            print("⊘ ESO已禁用，未保存扰动数据")
