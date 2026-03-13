
#!/usr/bin/env python3
# PID 数据可视化脚本（八桨 + 双舵机版本）
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.transform import Rotation as R

# Auto-detect log path based on script location
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
log_path = project_root / "log" / "csv" / "pid_underwater_data.csv"

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据
df = pd.read_csv(log_path)

# 创建输出目录
pic_dir = project_root / 'log' / 'pic'
pic_dir.mkdir(parents=True, exist_ok=True)

# 检查是否为悬停任务
goal_x_range = df['goal_x'].max() - df['goal_x'].min() if 'goal_x' in df.columns else 0
goal_y_range = df['goal_y'].max() - df['goal_y'].min() if 'goal_y' in df.columns else 0
is_hovering = (goal_x_range < 0.1) and (goal_y_range < 0.1)

# 计算位置误差
if 'goal_x' in df.columns:
    pos_error = np.sqrt((df['px'] - df['goal_x'])**2 + 
                        (df['py'] - df['goal_y'])**2 + 
                        (df['pz'] - df['goal_z'])**2)
else:
    pos_error = np.sqrt(df['px']**2 + df['py']**2 + (df['pz'] - 1.0)**2)


# ============================================================
# FIGURE 1: Position Tracking
# ============================================================
fig1 = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 6, figure=fig1)

# 上面3个子图：X, Y, Z位置跟踪（每个占2列）
ax1 = fig1.add_subplot(gs[0, 0:2])
ax1.plot(df['time'], df['px'], 'b-', label='Actual X', linewidth=2)
if 'goal_x' in df.columns:
    ax1.plot(df['time'], df['goal_x'], 'r--', label='Goal X', linewidth=1.5)
ax1.set_ylabel('X Position (m)', fontsize=11)
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('X Position Tracking', fontsize=12, fontweight='bold')

ax2 = fig1.add_subplot(gs[0, 2:4])
ax2.plot(df['time'], df['py'], 'b-', label='Actual Y', linewidth=2)
if 'goal_y' in df.columns:
    ax2.plot(df['time'], df['goal_y'], 'r--', label='Goal Y', linewidth=1.5)
ax2.set_ylabel('Y Position (m)', fontsize=11)
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Y Position Tracking', fontsize=12, fontweight='bold')

ax3 = fig1.add_subplot(gs[0, 4:6])
ax3.plot(df['time'], df['pz'], 'b-', label='Actual Z', linewidth=2)
if 'goal_z' in df.columns:
    ax3.plot(df['time'], df['goal_z'], 'r--', label='Goal Z', linewidth=1.5)
else:
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Goal Z (1.0m)', linewidth=1.5)
ax3.set_ylabel('Z Position (m)', fontsize=11)
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Z Position Tracking', fontsize=12, fontweight='bold')

# 下面2个子图居中：2D轨迹和位置误差（留出左右各1列的空白，实现居中效果）
ax4 = fig1.add_subplot(gs[1, 1:3])  # 下排居中偏左（从第2列到第4列）
if 'goal_x' in df.columns:
    ax4.plot(df['px'], df['py'], 'b-', linewidth=2, label='Trajectory')
    ax4.plot(df['goal_x'], df['goal_y'], 'r--', linewidth=1, alpha=0.6, label='Goal')
else:
    ax4.plot(df['px'], df['py'], 'b-', linewidth=2, label='Trajectory')
ax4.set_xlabel('X (m)', fontsize=11)
ax4.set_ylabel('Y (m)', fontsize=11)
ax4.legend()
ax4.axis('equal')
ax4.grid(True, alpha=0.3)
ax4.set_title('2D Trajectory (XY Plane)', fontsize=12, fontweight='bold')

ax5 = fig1.add_subplot(gs[1, 3:5])  # 下排居中偏右（从第4列到第6列）
ax5.plot(df['time'], pos_error, 'r-', linewidth=2)
ax5.set_ylabel('Position Error (m)', fontsize=11)
ax5.set_xlabel('Time (s)', fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.set_title('Position Tracking Error', fontsize=12, fontweight='bold')

plt.suptitle('Baseline Underwater - Position Tracking Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path1 = pic_dir / 'pid_underwater_position.png'
plt.savefig(pic_path1, dpi=150, bbox_inches='tight')
print(f'✓ Position plot saved: {pic_path1}')

# ============================================================
# FIGURE 2: Velocity
# ============================================================
fig2 = plt.figure(figsize=(16, 5))

# 5 线速度（如果有）
ax1 = plt.subplot(1, 3, 1)
if 'vx' in df.columns:
    ax1.plot(df['time'], df['vx'], label='Vx', linewidth=2)
    ax1.plot(df['time'], df['vy'], label='Vy', linewidth=2)
    ax1.plot(df['time'], df['vz'], label='Vz', linewidth=2)
    ax1.set_ylabel('Linear Velocity (m/s)', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.legend()
    ax1.set_title('Linear Velocity (Body Frame)', fontsize=12, fontweight='bold')
else:
    ax1.text(0.5, 0.5, 'Velocity data\nnot available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax1.transAxes)
    ax1.set_title('Linear Velocity', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 6 角速度（如果有）
ax2 = plt.subplot(1, 3, 2)
if 'wx' in df.columns:
    ax2.plot(df['time'], df['wx'], label='Wx (roll rate)', linewidth=2)
    ax2.plot(df['time'], df['wy'], label='Wy (pitch rate)', linewidth=2)
    ax2.plot(df['time'], df['wz'], label='Wz (yaw rate)', linewidth=2)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.legend()
    ax2.set_title('Angular Velocity (Body Frame)', fontsize=12, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Angular velocity\ndata not available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax2.transAxes)
    ax2.set_title('Angular Velocity', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 速度幅值
ax3 = plt.subplot(1, 3, 3)
if 'vx' in df.columns:
    vel_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
    ax3.plot(df['time'], vel_mag, 'g-', linewidth=2, label='Speed')
    ax3.set_ylabel('Speed (m/s)', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.legend()
    ax3.set_title('Total Speed (Velocity Magnitude)', fontsize=12, fontweight='bold')
else:
    ax3.text(0.5, 0.5, 'Velocity data\nnot available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax3.transAxes)
    ax3.set_title('Total Speed', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle('Baseline Underwater - Velocity Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path2 = pic_dir / 'pid_underwater_velocity.png'
plt.savefig(pic_path2, dpi=150, bbox_inches='tight')
print(f'✓ Velocity plot saved: {pic_path2}')

# ============================================================
# FIGURE 3: Control Inputs (Motors and Servos)
# ============================================================
fig3 = plt.figure(figsize=(14, 12))

# 8 电机转速（根据实际列名调整）
motor_cols = [col for col in df.columns if col.startswith('u') and col[1:].isdigit() and int(col[1:]) <= 8]

if len(motor_cols) >= 8:
    for idx, col in enumerate(sorted(motor_cols[:8]), 1):
        ax = plt.subplot(3, 3, idx)
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan']
        ax.plot(df['time'], df[col], linewidth=2, color=colors[idx-1], label=col)
        ax.set_ylabel(f'Motor {idx} (krpm)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Motor {idx} Speed', fontsize=11, fontweight='bold')
else:
    # 如果电机数据不足，尝试绘制可用的
    for idx, col in enumerate(motor_cols, 1):
        ax = plt.subplot(3, 3, idx)
        ax.plot(df['time'], df[col], linewidth=2, label=col)
        ax.set_ylabel(f'{col} (krpm)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{col} Speed', fontsize=11, fontweight='bold')

# 9 舵机角度
ax9 = plt.subplot(3, 3, 9)
if 'u9' in df.columns and 'u10' in df.columns:
    ax9.plot(df['time'], df['u9'], 'b-', label='Left Servo (u9)', linewidth=2)
    ax9.plot(df['time'], df['u10'], 'r-', label='Right Servo (u10)', linewidth=2)
    ax9.set_ylabel('Servo Angle (rad)', fontsize=10)
    ax9.set_xlabel('Time (s)', fontsize=11)
    ax9.legend(loc='best')
    ax9.set_title('Servo Angles', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3)

plt.suptitle('Baseline Underwater - Control Input Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path3 = pic_dir / 'pid_underwater_control.png'
plt.savefig(pic_path3, dpi=150, bbox_inches='tight')
print(f'✓ Control plot saved: {pic_path3}')

# ============================================================
# FIGURE 4: Attitude (Orientation)
# ============================================================
fig4 = plt.figure(figsize=(16, 5))

# 7 姿态角（如果有四元数）
if 'qw' in df.columns:
    quats = df[['qw', 'qx', 'qy', 'qz']].values
    euler_angles = R.from_quat(quats[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
    
    # Roll angle
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(df['time'], euler_angles[:, 0], 'b-', linewidth=2, label='Roll')
    ax1.set_ylabel('Roll Angle (deg)', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Roll Angle', fontsize=12, fontweight='bold')
    
    # Pitch angle
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(df['time'], euler_angles[:, 1], 'g-', linewidth=2, label='Pitch')
    ax2.set_ylabel('Pitch Angle (deg)', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Pitch Angle', fontsize=12, fontweight='bold')
    
    # Yaw angle
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(df['time'], euler_angles[:, 2], 'r-', linewidth=2, label='Yaw')
    ax3.set_ylabel('Yaw Angle (deg)', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Yaw Angle', fontsize=12, fontweight='bold')
else:
    ax1 = plt.subplot(1, 3, 1)
    ax1.text(0.5, 0.5, 'Attitude data\nnot available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax1.transAxes)
    ax1.set_title('Attitude (Roll/Pitch/Yaw)', fontsize=12, fontweight='bold')

plt.suptitle('Baseline Underwater - Performance Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path5 = pic_dir / 'pid_underwater_performance.png'
plt.savefig(pic_path5, dpi=150, bbox_inches='tight')
print(f'✓ Performance plot saved: {pic_path5}')

print("\n" + "="*50)
print("All plots generated successfully!")
print("="*50)

# Show all figures
plt.show()
