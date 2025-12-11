# 20250220 Wakkk
# Quadrotor SE3 Control Demo
import mujoco 
import mujoco.viewer as viewer 
import numpy as np
from nmpc_controller_underwater import NMPC_Controller
from trajectory_generator import TrajectoryGenerator
from eso_observer import ESO_Observer

# 新建NMPC控制器
controller = NMPC_Controller()

# 新建轨迹生成器
trajectory_gen = TrajectoryGenerator()

# 新建ESO扰动观测器
eso = ESO_Observer()

gravity = 9.8066        # 重力加速度 单位m/s^2
mass = 4.672            # 飞行器质量 单位kg
Ct = 0.1757            # 电机推力系数 (N/krpm^2)
Cd = 0.02          # 电机反扭系数 (Nm/krpm^2)

arm_length = 0.605/2.0  # 电机力臂长度 单位m
max_thrust = 17.75     # 单个电机最大推力 单位N (电机最大转速22krpm)
max_torque = 0.02  # 单个电机最大扭矩 单位Nm (电机最大转速22krpm)
left_servo_offset = -0.1
right_servo_offset = -0.25
# 仿真周期 100Hz 10ms 0.01s
dt = 0.01

# 根据电机转速计算电机推力
def calc_motor_force(krpm):
    global Ct
    return Ct * krpm**2

# 根据电机转速计算电机归一化输入
def calc_motor_input(krpm):
    if krpm > 22:
        krpm = 22
    elif krpm < 0:
        krpm = 0
    _force = calc_motor_force(krpm)
    _input = _force / max_thrust
    if _input > 1:
        _input = 1
    elif _input < 0:
        _input = 0
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

def control_callback(m, d):
    global log_count, gravity, mass, controller, trajectory_gen, eso, eso_enable

    pos = d.qpos[:3]        # [x, y, z]
    quat = d.qpos[3:7]      # [qw, qx, qy, qz]
    vel = d.qvel[:3]        # [vx, vy, vz]
    omega = d.qvel[3:6]     # [wx, wy, wz]

    current_state = np.concatenate([pos, quat, vel, omega])

    _pos = d.qpos
    _vel = d.qvel
    _sensor_data = d.sensordata
    gyro_x = _sensor_data[0]
    gyro_y = _sensor_data[1]
    gyro_z = _sensor_data[2]
    acc_x = _sensor_data[3]
    acc_y = _sensor_data[4]
    acc_z = _sensor_data[5]
    quat_w = _sensor_data[6]
    quat_x = _sensor_data[7]
    quat_y = _sensor_data[8]
    quat_z = _sensor_data[9]
    quat = np.array([quat_x, quat_y, quat_z, quat_w])  # x y z w
    omega = np.array([gyro_x, gyro_y, gyro_z])         # 角速度
    # 构建当前状态
    current_state = np.array([_pos[0], _pos[1], _pos[2], quat[3], quat[0], quat[1], quat[2], _vel[0], _vel[1], _vel[2], omega[0], omega[1], omega[2]])
    
    # 从轨迹生成器获取目标位置
    goal_position = trajectory_gen.get_reference(d.time)

    # NMPC Update（获取扰动补偿可选）
    k_yaw = 0.8
    # 获取扰动估计（可选用于前馈补偿）
    if eso_enable:
        disturbance = eso.get_all_disturbances(filtered=True)
    else:
        disturbance = None
    _dt, _control = controller.nmpc_position_control(current_state, goal_position, disturbance)
    # 计算实际的8个电机转速
    motor_speeds = np.array([
        _control[0] - k_yaw * _control[4],  # motor0: Front上,CW
        _control[1] + k_yaw * _control[4],  # motor1: Left上,CCW
        _control[2] - k_yaw * _control[4],  # motor2: Rear上,CW
        _control[3] + k_yaw * _control[4],  # motor3: Right上,CCW
        _control[0] + k_yaw * _control[4],  # motor4: Front下,CCW
        _control[1] - k_yaw * _control[4],  # motor5: Left下,CW
        _control[2] + k_yaw * _control[4],  # motor6: Rear下,CCW
        _control[3] - k_yaw * _control[4],  # motor7: Right下,CW
    ])
    
    # 应用电机控制
    for i in range(8):
        d.actuator(f'prop_motor{i}').ctrl[0] = calc_motor_input(motor_speeds[i])

    d.actuator('left_servo').ctrl[0] = 0.0 + left_servo_offset   # ~90 degrees
    d.actuator('right_servo').ctrl[0] = 0.0 + right_servo_offset
    
    # ========== 更新ESO观测器（仅在启用时） ==========
    if eso_enable:
        # 计算控制力和力矩（body frame）
        # 从电机转速计算总推力和力矩
        w_squared = motor_speeds**2  # 转速平方
        
        # 总推力 (body z轴方向)
        total_thrust = Ct * np.sum(w_squared)
        control_force_body = np.array([0.0, 0.0, total_thrust])
        
        # 控制力矩 [mx, my, mz]
        # 电机位置 (x, y) 相对于质心
        motor_positions = np.array([
            [arm_length, 0],      # motor0: Front
            [0, arm_length],      # motor1: Left
            [-arm_length, 0],     # motor2: Rear
            [0, -arm_length],     # motor3: Right
            [arm_length, 0],      # motor4: Front
            [0, arm_length],      # motor5: Left
            [-arm_length, 0],     # motor6: Rear
            [0, -arm_length],     # motor7: Right
        ])
        
        # Roll力矩 (绕x轴): 左右电机推力差
        mx = Ct * (motor_positions[1,1] * (w_squared[1] + w_squared[5]) - 
                   motor_positions[3,1] * (w_squared[3] + w_squared[7]))
        
        # Pitch力矩 (绕y轴): 前后电机推力差
        my = Ct * (-motor_positions[0,0] * (w_squared[0] + w_squared[4]) + 
                    motor_positions[2,0] * (w_squared[2] + w_squared[6]))
        
        # Yaw力矩 (绕z轴): 反扭力矩
        # CW: motor0, 2, 5, 7  (负)
        # CCW: motor1, 3, 4, 6 (正)
        mz = Cd * (-w_squared[0] + w_squared[1] - w_squared[2] + w_squared[3] + 
                    w_squared[4] - w_squared[5] + w_squared[6] - w_squared[7])
        
        control_torque_body = np.array([mx, my, mz])
        
        # 四元数转欧拉角
        qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(2*(qw*qy - qz*qx))
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        euler_angles = np.array([roll, pitch, yaw])
        
        # 更新ESO
        dt = m.opt.timestep
        eso.update(position=_pos[:3], 
                  velocity=_vel[:3],
                  euler_angles=euler_angles,
                  angular_velocity=omega,
                  control_force=control_force_body,
                  control_torque=control_torque_body,
                  dt=dt)
    
    log_count += 1
    if log_count >= 50:
        log_count = 0
        # 输出扰动估计（仅在ESO启用时）
        if eso_enable:
            dist = eso.get_all_disturbances(filtered=True)
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
                        help='飞行模式 (1-5)')
    parser.add_argument('--eso', dest='eso_enable', action='store_true', 
                        default=True, help='启用ESO扰动观测器（默认启用）')
    parser.add_argument('--no-eso', dest='eso_enable', action='store_false',
                        help='禁用ESO扰动观测器')
    
    args = parser.parse_args()
    
    # 设置全局ESO开关
    eso_enable = args.eso_enable
    
    print("="*80)
    print("🚁 八旋翼NMPC控制器 - 轨迹跟踪模式")
    print("="*80)
    print(f"\n【ESO扰动观测器】: {'✅ 启用' if eso_enable else '❌ 禁用'}")
    print("\n【选择飞行模式】")
    print("  1 - 悬停模式 (Hover at 1.0m)")
    print("  2 - 画圆模式 (Circle)")
    print("  3 - 画方形模式 (Square)")
    print("  4 - 连续爬升 (Climb: 0m → 2.5m, 从地面起飞) ⭐推荐")
    print("  5 - 自动演示 (Auto Demo)")
    
    # 如果有命令行参数，使用命令行参数，否则交互式输入
    if args.mode is not None:
        mode_choice = args.mode
    else:
        mode_choice = input("\n请选择模式 (1-5，默认1): ").strip() or "1"
    
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
