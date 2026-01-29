#!/usr/bin/env python3
# NMPC 数据可视化脚本 (包含ESO扰动估计)
import pandas as pd

# Auto-detect log path based on script location
import os
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
log_path = project_root / "log" / "csv" / "nmpc_data.csv"

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取CSV数据
df = pd.read_csv(log_path)

# 创建图形窗口
fig = plt.figure(figsize=(20, 12))
n_rows, n_cols = 4, 3  # 4行3列布局

# 1. 位置跟踪 (3个子图)
ax1 = plt.subplot(n_rows, n_cols, 1)
ax1.plot(df['time'], df['px'], 'b-', label='Actual X', linewidth=2)
ax1.plot(df['time'], df['goal_x'], 'r--', label='Goal X', linewidth=1.5)
ax1.set_ylabel('X Position (m)')
ax1.legend()
ax1.grid(True)
ax1.set_title('X Position Tracking')

ax2 = plt.subplot(n_rows, n_cols, 2)
ax2.plot(df['time'], df['py'], 'b-', label='Actual Y', linewidth=2)
ax2.plot(df['time'], df['goal_y'], 'r--', label='Goal Y', linewidth=1.5)
ax2.set_ylabel('Y Position (m)')
ax2.legend()
ax2.grid(True)
ax2.set_title('Y Position Tracking')

ax3 = plt.subplot(n_rows, n_cols, 3)
ax3.plot(df['time'], df['pz'], 'b-', label='Actual Z', linewidth=2)
ax3.plot(df['time'], df['goal_z'], 'r--', label='Goal Z', linewidth=1.5)
ax3.set_ylabel('Z Position (m)')
ax3.legend()
ax3.grid(True)
ax3.set_title('Z Position Tracking')

# 2. 2D轨迹 (XY平面) - 仅在非悬停时显示
ax4 = plt.subplot(n_rows, n_cols, 4)
# 检查是否为悬停任务（目标位置变化小于0.1m）
goal_x_range = df['goal_x'].max() - df['goal_x'].min()
goal_y_range = df['goal_y'].max() - df['goal_y'].min()
is_hovering = (goal_x_range < 0.1) and (goal_y_range < 0.1)

if not is_hovering:
    ax4.plot(df['px'], df['py'], 'b-', linewidth=2, label='Trajectory')
    ax4.plot(df['goal_x'], df['goal_y'], 'r--', linewidth=1, alpha=0.6, label='Goal')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.legend()
    ax4.set_title('2D Trajectory (XY Plane)')
    ax4.axis('equal')
else:
    ax4.text(0.5, 0.5, 'Hovering Task\n(No trajectory)', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax4.transAxes)
    ax4.set_title('2D Trajectory (XY Plane)')
ax4.grid(True)

# 3. 速度
ax5 = plt.subplot(n_rows, n_cols, 5)
ax5.plot(df['time'], df['vx'], label='Vx', linewidth=1.5)
ax5.plot(df['time'], df['vy'], label='Vy', linewidth=1.5)
ax5.plot(df['time'], df['vz'], label='Vz', linewidth=1.5)
ax5.set_ylabel('Velocity (m/s)')
ax5.legend()
ax5.grid(True)
ax5.set_title('Linear Velocity')

# 4. 角速度
ax6 = plt.subplot(n_rows, n_cols, 6)
ax6.plot(df['time'], df['wx'], label='Wx', linewidth=1.5)
ax6.plot(df['time'], df['wy'], label='Wy', linewidth=1.5)
ax6.plot(df['time'], df['wz'], label='Wz', linewidth=1.5)
ax6.set_ylabel('Angular Velocity (rad/s)')
ax6.legend()
ax6.grid(True)
ax6.set_title('Angular Velocity')

# 5. 四元数
ax7 = plt.subplot(n_rows, n_cols, 7)
ax7.plot(df['time'], df['qw'], label='qw', linewidth=1.5)
ax7.plot(df['time'], df['qx'], label='qx', linewidth=1.5)
ax7.plot(df['time'], df['qy'], label='qy', linewidth=1.5)
ax7.plot(df['time'], df['qz'], label='qz', linewidth=1.5)
ax7.set_ylabel('Quaternion')
ax7.legend()
ax7.grid(True)
ax7.set_title('Orientation (Quaternion)')

# 6. 控制输入 - 4个机臂转速
ax8 = plt.subplot(n_rows, n_cols, 8)
ax8.plot(df['time'], df['u_front'], label='Front', linewidth=1.5)
ax8.plot(df['time'], df['u_left'], label='Left', linewidth=1.5)
ax8.plot(df['time'], df['u_rear'], label='Rear', linewidth=1.5)
ax8.plot(df['time'], df['u_right'], label='Right', linewidth=1.5)
ax8.set_ylabel('Motor Speed (krpm)')
ax8.legend()
ax8.grid(True)
ax8.set_title('Control Input - Motor Speeds')

# 7. 偏航控制
ax9 = plt.subplot(n_rows, n_cols, 9)
ax9.plot(df['time'], df['u_yaw_bias'], 'g-', linewidth=2)
ax9.set_ylabel('Yaw Bias')
ax9.grid(True)
ax9.set_title('Yaw Control Bias')

# 8. 位置误差
ax10 = plt.subplot(n_rows, n_cols, 10)
pos_error = np.sqrt((df['px'] - df['goal_x'])**2 + 
                    (df['py'] - df['goal_y'])**2 + 
                    (df['pz'] - df['goal_z'])**2)
ax10.plot(df['time'], pos_error, 'r-', linewidth=2)
ax10.set_ylabel('Position Error (m)')
ax10.set_xlabel('Time (s)')
ax10.grid(True)
ax10.set_title('Position Tracking Error')

# 9. 求解时间
ax11 = plt.subplot(n_rows, n_cols, 11)
ax11.plot(df['time'], df['solve_time']*1000, 'b-', linewidth=1.5)
ax11.set_ylabel('Solve Time (ms)')
ax11.set_xlabel('Time (s)')
ax11.grid(True)
ax11.set_title('NMPC Solve Time')
ax11.axhline(y=10, color='r', linestyle='--', label='10ms (100Hz)')
ax11.legend()

# 10. 统计信息
ax12 = plt.subplot(n_rows, n_cols, 12)
ax12.axis('off')
stats_text = f"""
Statistics:
{'='*30}
Duration:    {df['time'].iloc[-1]:.2f} s
Data points: {len(df)}
Sample rate: {len(df)/df['time'].iloc[-1]:.1f} Hz

Position Error (m):
  Mean:      {pos_error.mean():.4f}
  Max:       {pos_error.max():.4f}
  Min:       {pos_error.min():.4f}

Solve Time (ms):
  Mean:      {df['solve_time'].mean()*1000:.2f}
  Max:       {df['solve_time'].max()*1000:.2f}
  Min:       {df['solve_time'].min()*1000:.2f}

Control Range (krpm):
  Front: [{df['u_front'].min():.2f}, {df['u_front'].max():.2f}]
  Left:  [{df['u_left'].min():.2f}, {df['u_left'].max():.2f}]
  Rear:  [{df['u_rear'].min():.2f}, {df['u_rear'].max():.2f}]
  Right: [{df['u_right'].min():.2f}, {df['u_right'].max():.2f}]
"""
ax12.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center',
          family='monospace', fontproperties={'weight': 'normal'})

plt.suptitle('NMPC Control Data Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存图片到log/pic目录
pic_dir = project_root / 'log' / 'pic'
pic_dir.mkdir(parents=True, exist_ok=True)
pic_path = pic_dir / 'nmpc_visualization.png'
plt.savefig(pic_path, dpi=150, bbox_inches='tight')
print(f"✓ 图片已保存: {pic_path}")

# 显示图片
plt.show()

