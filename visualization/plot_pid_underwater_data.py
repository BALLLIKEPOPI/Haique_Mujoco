#!/usr/bin/env python3
# PID 数据可视化脚本（八桨 + 双舵机版本）
import pandas as pd

# Auto-detect log path based on script location
import os
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
log_path = project_root / "log" / "csv" / "pid_underwater_data.csv"

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据
df = pd.read_csv(log_path)

# 检查CSV文件包含的列
print(f"CSV列名: {df.columns.tolist()}")

# 创建图形窗口
fig = plt.figure(figsize=(20, 12))
n_rows, n_cols = 4, 3  # 4行3列布局

# 1-3 位置跟踪
ax1 = plt.subplot(n_rows, n_cols, 1)
ax1.plot(df['time'], df['px'], 'b-', label='Actual X', linewidth=2)
if 'goal_x' in df.columns:
    ax1.plot(df['time'], df['goal_x'], 'r--', label='Goal X', linewidth=1.5)
ax1.set_ylabel('X (m)')
ax1.legend(); ax1.grid(True); ax1.set_title('X Position')

ax2 = plt.subplot(n_rows, n_cols, 2)
ax2.plot(df['time'], df['py'], 'b-', label='Actual Y', linewidth=2)
if 'goal_y' in df.columns:
    ax2.plot(df['time'], df['goal_y'], 'r--', label='Goal Y', linewidth=1.5)
ax2.set_ylabel('Y (m)')
ax2.legend(); ax2.grid(True); ax2.set_title('Y Position')

ax3 = plt.subplot(n_rows, n_cols, 3)
ax3.plot(df['time'], df['pz'], 'b-', label='Actual Z', linewidth=2)
if 'goal_z' in df.columns:
    ax3.plot(df['time'], df['goal_z'], 'r--', label='Goal Z', linewidth=1.5)
else:
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Goal Z (1.0m)', linewidth=1.5)
ax3.set_ylabel('Z (m)')
ax3.legend(); ax3.grid(True); ax3.set_title('Z Position')

# 4. 2D轨迹 (XY平面)
ax4 = plt.subplot(n_rows, n_cols, 4)
ax4.plot(df['px'], df['py'], 'b-', linewidth=2, label='Trajectory')
if 'goal_x' in df.columns and 'goal_y' in df.columns:
    ax4.plot(df['goal_x'], df['goal_y'], 'r--', linewidth=1, alpha=0.6, label='Goal')
ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
ax4.legend(); ax4.set_title('2D Trajectory (XY Plane)')
ax4.axis('equal'); ax4.grid(True)

# 5 线速度（如果有）
ax5 = plt.subplot(n_rows, n_cols, 5)
if 'vx' in df.columns:
    ax5.plot(df['time'], df['vx'], label='Vx', linewidth=1.5)
    ax5.plot(df['time'], df['vy'], label='Vy', linewidth=1.5)
    ax5.plot(df['time'], df['vz'], label='Vz', linewidth=1.5)
    ax5.set_ylabel('Velocity (m/s)')
    ax5.legend(); ax5.set_title('Linear Velocity')
else:
    ax5.text(0.5, 0.5, 'Velocity data\nnot available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax5.transAxes)
    ax5.set_title('Linear Velocity')
ax5.grid(True)

# 6 角速度（如果有）
ax6 = plt.subplot(n_rows, n_cols, 6)
if 'wx' in df.columns:
    ax6.plot(df['time'], df['wx'], label='Wx', linewidth=1.5)
    ax6.plot(df['time'], df['wy'], label='Wy', linewidth=1.5)
    ax6.plot(df['time'], df['wz'], label='Wz', linewidth=1.5)
    ax6.set_ylabel('Angular Vel (rad/s)')
    ax6.legend(); ax6.set_title('Angular Velocity')
else:
    ax6.text(0.5, 0.5, 'Angular velocity\ndata not available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax6.transAxes)
    ax6.set_title('Angular Velocity')
ax6.grid(True)

# 7 姿态角（如果有四元数）
ax7 = plt.subplot(n_rows, n_cols, 7)
if 'qw' in df.columns:
    from scipy.spatial.transform import Rotation as R
    quats = df[['qw', 'qx', 'qy', 'qz']].values
    euler_angles = R.from_quat(quats[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
    ax7.plot(df['time'], euler_angles[:, 0], label='Roll', linewidth=1.5)
    ax7.plot(df['time'], euler_angles[:, 1], label='Pitch', linewidth=1.5)
    ax7.plot(df['time'], euler_angles[:, 2], label='Yaw', linewidth=1.5)
    ax7.set_ylabel('Angle (deg)')
    ax7.legend(); ax7.set_title('Attitude (Roll/Pitch/Yaw)')
else:
    ax7.text(0.5, 0.5, 'Attitude data\nnot available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax7.transAxes)
    ax7.set_title('Attitude (Roll/Pitch/Yaw)')
ax7.grid(True)

# 8 电机转速（根据实际列名调整）
ax8 = plt.subplot(n_rows, n_cols, 8)
motor_cols = [col for col in df.columns if col.startswith('u') and col[1:].isdigit() and int(col[1:]) <= 8]
if motor_cols:
    for col in sorted(motor_cols):
        ax8.plot(df['time'], df[col], linewidth=1.2, label=col)
    ax8.set_ylabel('Motor Speed (krpm)')
    ax8.legend(ncol=2); ax8.set_title('Motor Speeds')
else:
    # 如果只有u1，单独绘制
    if 'u1' in df.columns:
        ax8.plot(df['time'], df['u1'], linewidth=1.5, label='u1')
        ax8.set_ylabel('Motor Speed (krpm)')
        ax8.legend(); ax8.set_title('Motor Speed (u1)')
ax8.grid(True)

# 9 舵机角度
ax9 = plt.subplot(n_rows, n_cols, 9)
if 'u9' in df.columns and 'u10' in df.columns:
    ax9.plot(df['time'], df['u9'], 'b-', label='Left Servo (u9)', linewidth=2)
    ax9.plot(df['time'], df['u10'], 'r-', label='Right Servo (u10)', linewidth=2)
    ax9.set_ylabel('Servo Angle (rad)')
    ax9.legend(loc='best'); ax9.set_title('Servo Angles')
ax9.grid(True)

# 10 Z轴位置详细
ax10 = plt.subplot(n_rows, n_cols, 10)
ax10.plot(df['time'], df['pz'], 'b-', linewidth=2, label='Actual Z')
if 'goal_z' in df.columns:
    ax10.plot(df['time'], df['goal_z'], 'r--', linewidth=1.5, label='Goal Z')
else:
    ax10.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5, label='Goal (1.0m)')
ax10.set_ylabel('Z (m)'); ax10.set_xlabel('Time (s)')
ax10.legend(); ax10.grid(True); ax10.set_title('Z Position Detail')

# 11 期望力/力矩（PID特有）
ax11 = plt.subplot(n_rows, n_cols, 11)
if 'des_fx' in df.columns:
    ax11_fx = ax11
    ax11_fx.plot(df['time'], df['des_fx'], 'b-', label='Desired Fx', linewidth=1.5)
    if 'des_fz' in df.columns:
        ax11_fx.plot(df['time'], df['des_fz'], 'r-', label='Desired Fz', linewidth=1.5)
    ax11_fx.set_ylabel('Force (N)'); ax11_fx.set_xlabel('Time (s)')
    ax11_fx.legend(loc='upper left'); ax11_fx.grid(True)
    
    if 'des_tx' in df.columns:
        ax11_tx = ax11_fx.twinx()
        ax11_tx.plot(df['time'], df['des_tx'], 'g--', label='Desired Tx', linewidth=1.5)
        ax11_tx.set_ylabel('Torque (Nm)', color='g')
        ax11_tx.tick_params(axis='y', labelcolor='g')
        ax11_tx.legend(loc='upper right')
    ax11.set_title('Desired Forces & Torque')
else:
    ax11.text(0.5, 0.5, 'Force/Torque data\nnot available', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax11.transAxes)
    ax11.set_title('Desired Forces & Torque')

# 12 统计信息
ax12 = plt.subplot(n_rows, n_cols, 12); ax12.axis('off')
stats_text = f"""
PID Controller Statistics:
{'='*35}
Duration:    {df['time'].iloc[-1]:.2f} s
Data points: {len(df)}
Sample rate: {len(df)/df['time'].iloc[-1]:.1f} Hz

Final Position:
  X:         {df['px'].iloc[-1]:.4f} m
  Y:         {df['py'].iloc[-1]:.4f} m
  Z:         {df['pz'].iloc[-1]:.4f} m

Position Range:
  X: [{df['px'].min():.3f}, {df['px'].max():.3f}]
  Y: [{df['py'].min():.3f}, {df['py'].max():.3f}]
  Z: [{df['pz'].min():.3f}, {df['pz'].max():.3f}]
"""

if 'qw' in df.columns:
    stats_text += f"""
Final Attitude (deg):
  Roll:      {euler_angles[-1, 0]:.2f}°
  Pitch:     {euler_angles[-1, 1]:.2f}°
  Yaw:       {euler_angles[-1, 2]:.2f}°
"""

ax12.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top', 
          family='monospace', fontproperties={'weight': 'normal'})

plt.tight_layout()

# 保存图片到log/pic目录
pic_dir = project_root / 'log' / 'pic'
pic_dir.mkdir(parents=True, exist_ok=True)
pic_path = pic_dir / 'pid_underwater_visualization.png'
plt.savefig(pic_path, dpi=150, bbox_inches='tight')
print(f'✓ 图片已保存: {pic_path}')
plt.show()
