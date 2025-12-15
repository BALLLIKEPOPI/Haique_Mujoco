#!/usr/bin/env python3
# NMPC 数据可视化脚本（八桨 + 双舵机版本，含 ESO 扰动）
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据
_df = pd.read_csv('./log/nmpc_data.csv')

# 列兼容性检查（缺失时给出友好提示）
required_cols = ['time', 'px', 'py', 'pz', 'goal_x', 'goal_y', 'goal_z',
                 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'qw', 'qx', 'qy', 'qz',
                 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u_alpha', 'u_beta', 'solve_time']
missing = [c for c in required_cols if c not in _df.columns]
if missing:
    raise ValueError(f"数据缺少字段: {missing}. 请确认已使用 nmpc_controller_underwater.py 生成日志。")

df = _df
has_disturbance = {'dist_fx', 'dist_fy', 'dist_fz', 'dist_mx', 'dist_my', 'dist_mz'}.issubset(df.columns)

# 布局：有扰动 6x3，共 18 个子图；无扰动 5x3，共 15 个子图
if has_disturbance:
    fig = plt.figure(figsize=(24, 16))
    n_rows, n_cols = 6, 3
else:
    fig = plt.figure(figsize=(22, 14))
    n_rows, n_cols = 5, 3

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

# 4 3D 轨迹
ax4 = plt.subplot(n_rows, n_cols, 4, projection='3d')
ax4.plot(df['px'], df['py'], df['pz'], 'b-', linewidth=2, label='Trajectory')
ax4.plot(df['goal_x'], df['goal_y'], df['goal_z'], 'r*', markersize=8, label='Goal')
ax4.set_xlabel('X (m)'); ax4.set_ylabel('Y (m)'); ax4.set_zlabel('Z (m)')
ax4.legend(); ax4.set_title('3D Trajectory')

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

# 9 舵机角
ax9 = plt.subplot(n_rows, n_cols, 9)
ax9.plot(df['time'], df['u_alpha'], label='alpha', linewidth=1.5)
ax9.plot(df['time'], df['u_beta'], label='beta', linewidth=1.5)
ax9.set_ylabel('Servo Angle (rad)')
ax9.legend(); ax9.grid(True); ax9.set_title('Servo Angles')

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

# 12 统计信息
ax12 = plt.subplot(n_rows, n_cols, 12); ax12.axis('off')
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
  u1: [{df['u1'].min():.2f}, {df['u1'].max():.2f}]
  u2: [{df['u2'].min():.2f}, {df['u2'].max():.2f}]
  u3: [{df['u3'].min():.2f}, {df['u3'].max():.2f}]
  u4: [{df['u4'].min():.2f}, {df['u4'].max():.2f}]
  u5: [{df['u5'].min():.2f}, {df['u5'].max():.2f}]
  u6: [{df['u6'].min():.2f}, {df['u6'].max():.2f}]
  u7: [{df['u7'].min():.2f}, {df['u7'].max():.2f}]
  u8: [{df['u8'].min():.2f}, {df['u8'].max():.2f}]
Servo Range (rad):
  alpha: [{df['u_alpha'].min():.3f}, {df['u_alpha'].max():.3f}]
  beta:  [{df['u_beta'].min():.3f}, {df['u_beta'].max():.3f}]
"""
ax12.text(0.05, 0.5, stats_text, fontsize=10, verticalalignment='center', family='monospace', fontproperties={'weight': 'normal'})

# ========== ESO 扰动 ========== 
if has_disturbance:
    ax13 = plt.subplot(n_rows, n_cols, 13)
    ax13.plot(df['time'], df['dist_fx'], 'b-', linewidth=1.3, label='Fx')
    ax13.plot(df['time'], df['dist_fy'], 'g-', linewidth=1.3, label='Fy')
    ax13.plot(df['time'], df['dist_fz'], 'r-', linewidth=1.3, label='Fz')
    ax13.set_ylabel('Dist Force (N)'); ax13.set_xlabel('Time (s)')
    ax13.legend(); ax13.grid(True); ax13.set_title('ESO Disturbance Force')

    ax14 = plt.subplot(n_rows, n_cols, 14)
    ax14.plot(df['time'], df['dist_mx'], 'b-', linewidth=1.3, label='Mx')
    ax14.plot(df['time'], df['dist_my'], 'g-', linewidth=1.3, label='My')
    ax14.plot(df['time'], df['dist_mz'], 'r-', linewidth=1.3, label='Mz')
    ax14.set_ylabel('Dist Torque (Nm)'); ax14.set_xlabel('Time (s)')
    ax14.legend(); ax14.grid(True); ax14.set_title('ESO Disturbance Torque')

    ax15 = plt.subplot(n_rows, n_cols, 15)
    force_mag = np.sqrt(df['dist_fx']**2 + df['dist_fy']**2 + df['dist_fz']**2)
    torque_mag = np.sqrt(df['dist_mx']**2 + df['dist_my']**2 + df['dist_mz']**2)
    ax15.plot(df['time'], force_mag, 'b-', linewidth=2, label='|F|')
    ax15_twin = ax15.twinx()
    ax15_twin.plot(df['time'], torque_mag, 'r-', linewidth=2, label='|M|')
    ax15.set_ylabel('Force (N)', color='b'); ax15_twin.set_ylabel('Torque (Nm)', color='r')
    ax15.set_xlabel('Time (s)'); ax15.grid(True); ax15.set_title('Disturbance Magnitude')
    ax15.legend(loc='upper left'); ax15_twin.legend(loc='upper right')

    ax16 = plt.subplot(n_rows, n_cols, 16); ax16.axis('off')
    dist_stats_text = f"""
ESO Disturbance Stats:
{'='*30}
Force (N):
  Fx: {np.nanmean(df['dist_fx']):.4f} ± {np.nanstd(df['dist_fx']):.4f}  Max|.| {np.nanmax(np.abs(df['dist_fx'])):.4f}
  Fy: {np.nanmean(df['dist_fy']):.4f} ± {np.nanstd(df['dist_fy']):.4f}  Max|.| {np.nanmax(np.abs(df['dist_fy'])):.4f}
  Fz: {np.nanmean(df['dist_fz']):.4f} ± {np.nanstd(df['dist_fz']):.4f}  Max|.| {np.nanmax(np.abs(df['dist_fz'])):.4f}

Torque (Nm):
  Mx: {np.nanmean(df['dist_mx']):.4f} ± {np.nanstd(df['dist_mx']):.4f}  Max|.| {np.nanmax(np.abs(df['dist_mx'])):.4f}
  My: {np.nanmean(df['dist_my']):.4f} ± {np.nanstd(df['dist_my']):.4f}  Max|.| {np.nanmax(np.abs(df['dist_my'])):.4f}
  Mz: {np.nanmean(df['dist_mz']):.4f} ± {np.nanstd(df['dist_mz']):.4f}  Max|.| {np.nanmax(np.abs(df['dist_mz'])):.4f}
"""
    ax16.text(0.05, 0.5, dist_stats_text, fontsize=9, verticalalignment='center', family='monospace', fontproperties={'weight': 'normal'})

    ax17 = plt.subplot(n_rows, n_cols, 17)
    valid_mask_xy = ~(np.isnan(df['dist_fx']) | np.isnan(df['dist_fy']))
    if valid_mask_xy.sum() > 0:
        scatter = ax17.scatter(df['dist_fx'][valid_mask_xy], df['dist_fy'][valid_mask_xy], c=df['time'][valid_mask_xy], cmap='viridis', s=6, alpha=0.5)
        plt.colorbar(scatter, ax=ax17, label='Time (s)')
    ax17.set_xlabel('Fx (N)'); ax17.set_ylabel('Fy (N)'); ax17.grid(True, alpha=0.3); ax17.set_title('Disturbance XY')

    ax18 = plt.subplot(n_rows, n_cols, 18)
    force_mag = np.sqrt(df['dist_fx']**2 + df['dist_fy']**2 + df['dist_fz']**2)
    valid_mask_fz = ~(np.isnan(df['dist_fz']) | np.isnan(force_mag))
    if valid_mask_fz.sum() > 0:
        ax18.hist2d(df['dist_fz'][valid_mask_fz], force_mag[valid_mask_fz], bins=30, cmap='YlOrRd')
        ax18.set_xlabel('Fz (N)'); ax18.set_ylabel('|F| (N)'); ax18.set_title('Fz vs |F|')
    else:
        ax18.text(0.5, 0.5, 'No valid data', ha='center', va='center')
        ax18.set_title('Fz vs |F| (No Data)')

if has_disturbance:
    plt.suptitle('Underwater NMPC + ESO (8 Motors + 2 Servos)', fontsize=16, fontweight='bold')
else:
    plt.suptitle('Underwater NMPC (8 Motors + 2 Servos)', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('nmpc_underwater_visualization.png', dpi=150, bbox_inches='tight')
print('✓ 图片已保存: nmpc_underwater_visualization.png')
plt.show()
