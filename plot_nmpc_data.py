#!/usr/bin/env python3
# NMPC 数据可视化脚本 (包含ESO扰动估计)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取CSV数据
df = pd.read_csv('./log/nmpc_data.csv')

# 检查是否有扰动数据列
has_disturbance = 'dist_fx' in df.columns

# 创建大图窗口 (根据是否有扰动数据调整布局)
if has_disturbance:
    fig = plt.figure(figsize=(24, 16))  # 更大的窗口容纳扰动图
else:
    fig = plt.figure(figsize=(20, 12))

# 根据是否有扰动数据选择布局
if has_disturbance:
    n_rows, n_cols = 6, 3  # 6行3列布局
else:
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

# 2. 3D轨迹
ax4 = plt.subplot(n_rows, n_cols, 4, projection='3d')
ax4.plot(df['px'], df['py'], df['pz'], 'b-', linewidth=2, label='Trajectory')
ax4.plot(df['goal_x'], df['goal_y'], df['goal_z'], 'r*', markersize=10, label='Goal')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_zlabel('Z (m)')
ax4.legend()
ax4.set_title('3D Trajectory')

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

# ========== ESO扰动估计图表（如果有数据） ==========
if has_disturbance:
    # 11. 扰动力 Fx, Fy, Fz
    ax13 = plt.subplot(n_rows, n_cols, 13)
    ax13.plot(df['time'], df['dist_fx'], 'b-', linewidth=1.5, label='Fx')
    ax13.plot(df['time'], df['dist_fy'], 'g-', linewidth=1.5, label='Fy')
    ax13.plot(df['time'], df['dist_fz'], 'r-', linewidth=1.5, label='Fz')
    ax13.set_ylabel('Disturbance Force (N)')
    ax13.set_xlabel('Time (s)')
    ax13.legend()
    ax13.grid(True)
    ax13.set_title('ESO - Disturbance Force')
    
    # 12. 扰动力矩 Mx, My, Mz
    ax14 = plt.subplot(n_rows, n_cols, 14)
    ax14.plot(df['time'], df['dist_mx'], 'b-', linewidth=1.5, label='Mx (Roll)')
    ax14.plot(df['time'], df['dist_my'], 'g-', linewidth=1.5, label='My (Pitch)')
    ax14.plot(df['time'], df['dist_mz'], 'r-', linewidth=1.5, label='Mz (Yaw)')
    ax14.set_ylabel('Disturbance Torque (Nm)')
    ax14.set_xlabel('Time (s)')
    ax14.legend()
    ax14.grid(True)
    ax14.set_title('ESO - Disturbance Torque')
    
    # 13. 扰动幅值
    ax15 = plt.subplot(n_rows, n_cols, 15)
    force_mag = np.sqrt(df['dist_fx']**2 + df['dist_fy']**2 + df['dist_fz']**2)
    torque_mag = np.sqrt(df['dist_mx']**2 + df['dist_my']**2 + df['dist_mz']**2)
    ax15.plot(df['time'], force_mag, 'b-', linewidth=2, label='Force Magnitude')
    ax15_twin = ax15.twinx()
    ax15_twin.plot(df['time'], torque_mag, 'r-', linewidth=2, label='Torque Magnitude')
    ax15.set_ylabel('Force (N)', color='b')
    ax15_twin.set_ylabel('Torque (Nm)', color='r')
    ax15.set_xlabel('Time (s)')
    ax15.tick_params(axis='y', labelcolor='b')
    ax15_twin.tick_params(axis='y', labelcolor='r')
    ax15.grid(True)
    ax15.set_title('ESO - Total Disturbance Magnitude')
    ax15.legend(loc='upper left')
    ax15_twin.legend(loc='upper right')
    
    # 14. 扰动统计
    ax16 = plt.subplot(n_rows, n_cols, 16)
    ax16.axis('off')
    # 使用nanmean和nanstd处理可能的NaN值
    dist_stats_text = f"""
ESO Disturbance Stats:
{'='*30}
Force (N):
  Fx: {np.nanmean(df['dist_fx']):.4f} ± {np.nanstd(df['dist_fx']):.4f}
      Max: {np.nanmax(np.abs(df['dist_fx'])):.4f}
  Fy: {np.nanmean(df['dist_fy']):.4f} ± {np.nanstd(df['dist_fy']):.4f}
      Max: {np.nanmax(np.abs(df['dist_fy'])):.4f}
  Fz: {np.nanmean(df['dist_fz']):.4f} ± {np.nanstd(df['dist_fz']):.4f}
      Max: {np.nanmax(np.abs(df['dist_fz'])):.4f}

Torque (Nm):
  Mx: {np.nanmean(df['dist_mx']):.4f} ± {np.nanstd(df['dist_mx']):.4f}
      Max: {np.nanmax(np.abs(df['dist_mx'])):.4f}
  My: {np.nanmean(df['dist_my']):.4f} ± {np.nanstd(df['dist_my']):.4f}
      Max: {np.nanmax(np.abs(df['dist_my'])):.4f}
  Mz: {np.nanmean(df['dist_mz']):.4f} ± {np.nanstd(df['dist_mz']):.4f}
      Max: {np.nanmax(np.abs(df['dist_mz'])):.4f}
"""
    ax16.text(0.05, 0.5, dist_stats_text, fontsize=9, verticalalignment='center',
              family='monospace', fontproperties={'weight': 'normal'})
    
    # 15-18: XY, XZ, YZ平面的扰动力分布
    ax17 = plt.subplot(n_rows, n_cols, 17)
    # 过滤NaN值
    valid_mask_scatter = ~(np.isnan(df['dist_fx']) | np.isnan(df['dist_fy']))
    if valid_mask_scatter.sum() > 0:
        scatter = ax17.scatter(df['dist_fx'][valid_mask_scatter], df['dist_fy'][valid_mask_scatter], 
                               c=df['time'][valid_mask_scatter], cmap='viridis', s=5, alpha=0.5)
        plt.colorbar(scatter, ax=ax17, label='Time (s)')
    ax17.set_xlabel('Fx (N)')
    ax17.set_ylabel('Fy (N)')
    ax17.grid(True, alpha=0.3)
    ax17.set_title('Disturbance Force - XY Plane')
    
    ax18 = plt.subplot(n_rows, n_cols, 18)
    # 过滤NaN值
    valid_mask = ~(np.isnan(df['dist_fz']) | np.isnan(force_mag))
    if valid_mask.sum() > 0:
        ax18.hist2d(df['dist_fz'][valid_mask], force_mag[valid_mask], bins=30, cmap='YlOrRd')
        ax18.set_xlabel('Fz (N)')
        ax18.set_ylabel('Total Force Magnitude (N)')
        ax18.set_title('Fz vs Total Force Distribution')
    else:
        ax18.text(0.5, 0.5, 'No valid data', ha='center', va='center')
        ax18.set_title('Fz vs Total Force Distribution (No Data)')
    
    print(f"✓ ESO扰动数据已包含在可视化中")

if has_disturbance:
    plt.suptitle('NMPC Control & ESO Disturbance Data Visualization', fontsize=16, fontweight='bold')
else:
    plt.suptitle('NMPC Control Data Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存图片
plt.savefig('nmpc_visualization.png', dpi=150, bbox_inches='tight')
print(f"✓ 图片已保存: nmpc_visualization.png")

# 显示图片
plt.show()

