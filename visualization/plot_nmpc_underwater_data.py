#!/usr/bin/env python3
# NMPC 数据可视化脚本（八桨 + 双舵机版本，含 ESO 扰动）
import pandas as pd

# Auto-detect log path based on script location
import os
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
log_path = project_root / "log" / "csv" / "nmpc_underwater_data.csv"

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据
df = pd.read_csv(log_path)

# 创建图形窗口
fig = plt.figure(figsize=(20, 12))
n_rows, n_cols = 4, 3  # 4行3列布局

# 1-3 位置跟踪
ax1 = plt.subplot(n_rows, n_cols, 1)
ax1.plot(df['time'], df['px'], 'b-', label='Actual X', linewidth=2)
ax1.plot(df['time'], df['goal_x'], 'r--', label='Goal X', linewidth=1.5)
ax1.set_ylabel('X (m)')
ax1.legend(); ax1.grid(True); ax1.set_title('X Position')

ax2 = plt.subplot(n_rows, n_cols, 2)
ax2.plot(df['time'], df['py'], 'b-', label='Actual Y', linewidth=2)
ax2.plot(df['time'], df['goal_y'], 'r--', label='Goal Y', linewidth=1.5)
ax2.set_ylabel('Y (m)')
ax2.legend(); ax2.grid(True); ax2.set_title('Y Position')

ax3 = plt.subplot(n_rows, n_cols, 3)
ax3.plot(df['time'], df['pz'], 'b-', label='Actual Z', linewidth=2)
ax3.plot(df['time'], df['goal_z'], 'r--', label='Goal Z', linewidth=1.5)
ax3.set_ylabel('Z (m)')
ax3.legend(); ax3.grid(True); ax3.set_title('Z Position')

# 4. 2D轨迹 (XY平面) - 仅在非悬停时显示
ax4 = plt.subplot(n_rows, n_cols, 4)
# 检查是否为悬停任务（目标位置变化小于0.1m）
goal_x_range = df['goal_x'].max() - df['goal_x'].min()
goal_y_range = df['goal_y'].max() - df['goal_y'].min()
is_hovering = (goal_x_range < 0.1) and (goal_y_range < 0.1)

if not is_hovering:
    ax4.plot(df['px'], df['py'], 'b-', linewidth=2, label='Trajectory')
    ax4.plot(df['goal_x'], df['goal_y'], 'r--', linewidth=1, alpha=0.6, label='Goal')
    ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)')
    ax4.legend(); ax4.set_title('2D Trajectory (XY Plane)')
    ax4.axis('equal')
else:
    ax4.text(0.5, 0.5, 'Hovering Task\n(No trajectory)', 
             ha='center', va='center', fontsize=12, color='gray',
             transform=ax4.transAxes)
    ax4.set_title('2D Trajectory (XY Plane)')
ax4.grid(True)

# 5 线速度
ax5 = plt.subplot(n_rows, n_cols, 5)
ax5.plot(df['time'], df['vx'], label='Vx', linewidth=1.5)
ax5.plot(df['time'], df['vy'], label='Vy', linewidth=1.5)
ax5.plot(df['time'], df['vz'], label='Vz', linewidth=1.5)
ax5.set_ylabel('Velocity (m/s)')
ax5.legend(); ax5.grid(True); ax5.set_title('Linear Velocity')

# 6 角速度
ax6 = plt.subplot(n_rows, n_cols, 6)
ax6.plot(df['time'], df['wx'], label='Wx', linewidth=1.5)
ax6.plot(df['time'], df['wy'], label='Wy', linewidth=1.5)
ax6.plot(df['time'], df['wz'], label='Wz', linewidth=1.5)
ax6.set_ylabel('Angular Vel (rad/s)')
ax6.legend(); ax6.grid(True); ax6.set_title('Angular Velocity')

# 7 四元数
ax7 = plt.subplot(n_rows, n_cols, 7)
ax7.plot(df['time'], df['qw'], label='qw', linewidth=1.0)
ax7.plot(df['time'], df['qx'], label='qx', linewidth=1.0)
ax7.plot(df['time'], df['qy'], label='qy', linewidth=1.0)
ax7.plot(df['time'], df['qz'], label='qz', linewidth=1.0)
ax7.set_ylabel('Quaternion')
ax7.legend(); ax7.grid(True); ax7.set_title('Orientation')

# 8 八个电机转速
ax8 = plt.subplot(n_rows, n_cols, 8)
for i in range(1, 9):
    ax8.plot(df['time'], df[f'u{i}'], linewidth=1.2, label=f'u{i}')
ax8.set_ylabel('Motor Speed (krpm)')
ax8.legend(ncol=2); ax8.grid(True); ax8.set_title('Motor Speeds (u1~u8)')

# 9 舵机角（细实线：NMPC原始，粗虚线：滤波后）
ax9 = plt.subplot(n_rows, n_cols, 9)
# NMPC原始输出（滤波前）- 使用细线和较低透明度
ax9.plot(df['time'], df['u_alpha'], 'b-', label='alpha (raw)', linewidth=1.0, alpha=0.5)
ax9.plot(df['time'], df['u_beta'], 'r-', label='beta (raw)', linewidth=1.0, alpha=0.5)
# 滤波后的舵机角度（实际执行）- 使用粗虚线和高对比度
if 'u_alpha_filt' in df.columns and 'u_beta_filt' in df.columns:
    ax9.plot(df['time'], df['u_alpha_filt'], 'b--', label='alpha (filtered)', linewidth=2.5, alpha=1.0, dashes=(5, 3))
    ax9.plot(df['time'], df['u_beta_filt'], 'r--', label='beta (filtered)', linewidth=2.5, alpha=1.0, dashes=(5, 3))
ax9.set_ylabel('Servo Angle (rad)')
ax9.legend(loc='best', fontsize=9); ax9.grid(True, alpha=0.3); ax9.set_title('Servo Angles (Thin: Raw, Bold Dash: Filtered)')

# 10 位置误差
ax10 = plt.subplot(n_rows, n_cols, 10)
pos_error = np.sqrt((df['px'] - df['goal_x'])**2 + (df['py'] - df['goal_y'])**2 + (df['pz'] - df['goal_z'])**2)
ax10.plot(df['time'], pos_error, 'r-', linewidth=2)
ax10.set_ylabel('Pos Error (m)'); ax10.set_xlabel('Time (s)')
ax10.grid(True); ax10.set_title('Position Error')

# 11 求解时间
ax11 = plt.subplot(n_rows, n_cols, 11)
ax11.plot(df['time'], df['solve_time']*1000, 'b-', linewidth=1.5)
ax11.set_ylabel('Solve Time (ms)'); ax11.set_xlabel('Time (s)')
ax11.grid(True); ax11.set_title('NMPC Solve Time')
ax11.axhline(y=10, color='r', linestyle='--', label='10ms (100Hz)'); ax11.legend()

# 12 统计信息（两列显示）
ax12 = plt.subplot(n_rows, n_cols, 12); ax12.axis('off')
# 左列
stats_text_left = f"""
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
"""

ax12.text(0.05, 0.95, stats_text_left, fontsize=10, verticalalignment='top', family='monospace', fontproperties={'weight': 'normal'})
# ax12.text(0.52, 0.95, stats_text_right, fontsize=10, verticalalignment='top', family='monospace', fontproperties={'weight': 'normal'})

# 保存图片到log/pic目录
from pathlib import Path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
pic_dir = project_root / 'log' / 'pic'
pic_dir.mkdir(parents=True, exist_ok=True)
pic_path = pic_dir / 'nmpc_underwater_visualization.png'
plt.savefig(pic_path, dpi=150, bbox_inches='tight')
print(f'✓ 图片已保存: {pic_path}')
plt.show()
