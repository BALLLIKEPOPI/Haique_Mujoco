#!/usr/bin/env python3
"""
电机失能仿真数据可视化脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_motor_failure_results(filename='motor_failure_data.npz', save_path='motor_failure_visualization.png'):
    """可视化电机失能仿真结果
    
    Args:
        filename: 数据文件路径
        save_path: 图片保存路径
    """
    # 加载数据
    data = np.load(filename)
    time = data['time']
    pos = data['pos']
    euler = data['euler']
    vel = data['vel']
    omega = data['omega']
    control = data['control']
    motor_inputs = data['motor_inputs']  # 归一化输入[0,1]
    motor_failure = data['motor_failure']
    motor_power_ratios = data['motor_power_ratios']  # 电机功率比例[8]
    
    print(f"Loaded data: {filename}")
    print(f"  Duration: {time[-1]:.2f}s")
    print(f"  Data points: {len(time)}")
    
    # 找到失能时刻和降级电机
    failure_indices = np.where(motor_failure)[0]
    motor_names = [
        'M0_Front_Up_CW', 'M1_Left_Up_CCW', 'M2_Rear_Up_CW', 'M3_Right_Up_CCW',
        'M4_Front_Low_CCW', 'M5_Left_Low_CW', 'M6_Rear_Low_CCW', 'M7_Right_Low_CW'
    ]
    
    if len(failure_indices) > 0:
        failure_time = time[failure_indices[0]]
        failure_idx = failure_indices[0]
        # 获取失能时刻的电机功率比例
        power_ratios_at_failure = motor_power_ratios[failure_idx]
        # 找到所有降级的电机（功率<100%）
        degraded_motors = np.where(power_ratios_at_failure < 1.0)[0]
        if len(degraded_motors) > 0:
            degradation_info = [f"{motor_names[i]}={power_ratios_at_failure[i]*100:.0f}%" 
                               for i in degraded_motors]
            print(f"  Motor degradation: {', '.join(degradation_info)} at t={failure_time:.2f}s")
        else:
            print(f"  No motor degradation")
    else:
        failure_time = None
        failure_idx = -1
        degraded_motors = []
        power_ratios_at_failure = np.ones(8)
        print(f"  No motor failure")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 颜色
    color_normal = '#2E86AB'
    color_failure = '#A23B72'
    
    # 1. 位置 (3个子图)
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(time, pos[:, i], color=color_normal, linewidth=1.5, label='Position')
        if failure_time is not None:
            ax.axvline(failure_time, color=color_failure, linestyle='--', 
                      linewidth=2, alpha=0.7, label='Motor Failure')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Position {label} (m)')
        ax.set_title(f'Position {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # 2. 欧拉角 (3个子图)
    euler_labels = ['Roll', 'Pitch', 'Yaw']
    for i, label in enumerate(euler_labels):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(time, euler[:, i], color=color_normal, linewidth=1.5, label=label)
        if failure_time is not None:
            ax.axvline(failure_time, color=color_failure, linestyle='--', 
                      linewidth=2, alpha=0.7, label='Motor Failure')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{label} (deg)')
        ax.set_title(f'Attitude: {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # 3. 速度 (3个子图)
    for i, label in enumerate(['Vx', 'Vy', 'Vz']):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(time, vel[:, i], color=color_normal, linewidth=1.5, label='Velocity')
        if failure_time is not None:
            ax.axvline(failure_time, color=color_failure, linestyle='--', 
                      linewidth=2, alpha=0.7, label='Motor Failure')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Velocity {label} (m/s)')
        ax.set_title(f'Velocity {label}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    
    # 4. 电机转速 (8个电机，绘制在一个图中)
    ax = fig.add_subplot(gs[3, 0])
    motor_labels = [
        'M0_F_Up_CW', 'M1_L_Up_CCW', 'M2_R_Up_CW', 'M3_Ri_Up_CCW',
        'M4_F_Low_CCW', 'M5_L_Low_CW', 'M6_R_Low_CCW', 'M7_Ri_Low_CW'
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    for i in range(8):
        # 如果该电机降级，使用特殊颜色和标记
        if failure_time is not None and i in degraded_motors:
            power_pct = power_ratios_at_failure[i] * 100
            if power_pct == 0:
                label = f'{motor_labels[i]} (FAILED)'
            else:
                label = f'{motor_labels[i]} ({power_pct:.0f}%)'
            ax.plot(time, motor_inputs[:, i], color=color_failure, 
                   linewidth=2.5, label=label, linestyle='--', alpha=0.9)
        else:
            ax.plot(time, motor_inputs[:, i], color=colors[i], 
                   linewidth=1.2, label=motor_labels[i], alpha=0.7)
    
    if failure_time is not None:
        ax.axvline(failure_time, color=color_failure, linestyle='--', 
                  linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Motor Input (normalized [0,1])')
    ax.set_title('Motor Inputs')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2, fontsize=8)
    
    # 5. 角速度
    ax = fig.add_subplot(gs[3, 1])
    omega_labels = ['ωx', 'ωy', 'ωz']
    for i, label in enumerate(omega_labels):
        ax.plot(time, omega[:, i], linewidth=1.5, label=label)
    if failure_time is not None:
        ax.axvline(failure_time, color=color_failure, linestyle='--', 
                  linewidth=2, alpha=0.7, label='Motor Failure')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Angular Velocity')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # 6. NMPC控制输出
    ax = fig.add_subplot(gs[3, 2])
    control_labels = ['w_front', 'w_left', 'w_rear', 'w_right', 'yaw_bias']
    for i, label in enumerate(control_labels):
        ax.plot(time, control[:, i], linewidth=1.5, label=label, alpha=0.7)
    if failure_time is not None:
        ax.axvline(failure_time, color=color_failure, linestyle='--', 
                  linewidth=2, alpha=0.7, label='Motor Failure')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Control Output (krpm)')
    ax.set_title('NMPC Control Output')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    
    # 总标题
    if failure_time is not None and len(degraded_motors) > 0:
        degradation_str = ', '.join([f"{motor_names[i]}({power_ratios_at_failure[i]*100:.0f}%)" 
                                     for i in degraded_motors])
        fig.suptitle(f'Motor Failure Simulation - {degradation_str} at t={failure_time:.2f}s',
                    fontsize=14, fontweight='bold')
    else:
        fig.suptitle('Motor Failure Simulation - Normal Operation',
                    fontsize=14, fontweight='bold')
    
    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    
    # 显示
    plt.show()


def print_statistics(filename='motor_failure_data.npz'):
    """打印统计信息"""
    data = np.load(filename)
    time = data['time']
    pos = data['pos']
    euler = data['euler']
    motor_failure = data['motor_failure']
    
    # 找到失能时刻
    failure_indices = np.where(motor_failure)[0]
    if len(failure_indices) > 0:
        failure_idx = failure_indices[0]
        failure_time = time[failure_idx]
        
        # 失能前后统计
        pre_failure = slice(max(0, failure_idx - 200), failure_idx)
        post_failure = slice(failure_idx, min(len(time), failure_idx + 200))
        
        print("\n" + "="*60)
        print("Statistical Analysis")
        print("="*60)
        
        print(f"\nBefore Failure (last 1s):")
        print(f"  Position std: X={np.std(pos[pre_failure, 0]):.4f}m, "
              f"Y={np.std(pos[pre_failure, 1]):.4f}m, "
              f"Z={np.std(pos[pre_failure, 2]):.4f}m")
        print(f"  Attitude std: Roll={np.std(euler[pre_failure, 0]):.2f}°, "
              f"Pitch={np.std(euler[pre_failure, 1]):.2f}°, "
              f"Yaw={np.std(euler[pre_failure, 2]):.2f}°")
        
        print(f"\nAfter Failure (next 1s):")
        print(f"  Position std: X={np.std(pos[post_failure, 0]):.4f}m, "
              f"Y={np.std(pos[post_failure, 1]):.4f}m, "
              f"Z={np.std(pos[post_failure, 2]):.4f}m")
        print(f"  Attitude std: Roll={np.std(euler[post_failure, 0]):.2f}°, "
              f"Pitch={np.std(euler[post_failure, 1]):.2f}°, "
              f"Yaw={np.std(euler[post_failure, 2]):.2f}°")
        
        print(f"\nMax deviation after failure:")
        post_all = slice(failure_idx, len(time))
        print(f"  Position: X={np.max(np.abs(pos[post_all, 0])):.3f}m, "
              f"Y={np.max(np.abs(pos[post_all, 1])):.3f}m, "
              f"Z={np.max(np.abs(pos[post_all, 2] - 1.0)):.3f}m")
        print(f"  Attitude: Roll={np.max(np.abs(euler[post_all, 0])):.1f}°, "
              f"Pitch={np.max(np.abs(euler[post_all, 1])):.1f}°, "
              f"Yaw={np.max(np.abs(euler[post_all, 2])):.1f}°")
        
        print("="*60 + "\n")


if __name__ == '__main__':
    filename = 'motor_failure_data.npz'
    
    # 打印统计信息
    print_statistics(filename)
    
    # 可视化
    plot_motor_failure_results(filename)
