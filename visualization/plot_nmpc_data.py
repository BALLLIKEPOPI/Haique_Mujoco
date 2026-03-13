#!/usr/bin/env python3
# NMPC Data Visualization Script
import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Auto-detect log path based on script location
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
log_path = project_root / "log" / "csv" / "nmpc_data.csv"

# Read CSV data
df = pd.read_csv(log_path)

# Create output directory
pic_dir = project_root / 'log' / 'pic'
pic_dir.mkdir(parents=True, exist_ok=True)

# Check if hovering task
goal_x_range = df['goal_used_x'].max() - df['goal_used_x'].min()
goal_y_range = df['goal_used_y'].max() - df['goal_used_y'].min()
is_hovering = (goal_x_range < 0.1) and (goal_y_range < 0.1)

# Calculate position error
pos_error = np.sqrt((df['px'] - df['goal_used_x'])**2 + 
                    (df['py'] - df['goal_used_y'])**2 + 
                    (df['pz'] - df['goal_used_z'])**2)

# ============================================================
# FIGURE 1: Position Tracking
# ============================================================
fig1 = plt.figure(figsize=(14, 10))

# X Position
ax1 = plt.subplot(2, 2, 1)
ax1.plot(df['time'], df['px'], 'b-', label='Actual X', linewidth=2)
ax1.plot(df['time'], df['goal_used_x'], 'r--', label='Goal X', linewidth=1.5)
ax1.set_ylabel('X Position (m)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('X Position Tracking', fontsize=12, fontweight='bold')

# Y Position
ax2 = plt.subplot(2, 2, 2)
ax2.plot(df['time'], df['py'], 'b-', label='Actual Y', linewidth=2)
ax2.plot(df['time'], df['goal_used_y'], 'r--', label='Goal Y', linewidth=1.5)
ax2.set_ylabel('Y Position (m)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Y Position Tracking', fontsize=12, fontweight='bold')

# Z Position
ax3 = plt.subplot(2, 2, 3)
ax3.plot(df['time'], df['pz'], 'b-', label='Actual Z', linewidth=2)
ax3.plot(df['time'], df['goal_used_z'], 'r--', label='Goal Z', linewidth=1.5)
ax3.set_ylabel('Z Position (m)', fontsize=11)
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Z Position Tracking', fontsize=12, fontweight='bold')

# 2D Trajectory (XY plane) / Position Error
ax4 = plt.subplot(2, 2, 4)
if not is_hovering:
    ax4.plot(df['px'], df['py'], 'b-', linewidth=2, label='Trajectory')
    ax4.plot(df['goal_used_x'], df['goal_used_y'], 'r--', linewidth=1, alpha=0.6, label='Goal')
    ax4.set_xlabel('X (m)', fontsize=11)
    ax4.set_ylabel('Y (m)', fontsize=11)
    ax4.legend()
    ax4.axis('equal')
    ax4.set_title('2D Trajectory (XY Plane)', fontsize=12, fontweight='bold')
else:
    ax4.plot(df['time'], pos_error, 'r-', linewidth=2)
    ax4.set_ylabel('Position Error (m)', fontsize=11)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_title('Position Tracking Error', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('NMPC - Position Tracking Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path1 = pic_dir / 'nmpc_position.png'
plt.savefig(pic_path1, dpi=150, bbox_inches='tight')
print(f"✓ Position plot saved: {pic_path1}")

# ============================================================
# FIGURE 2: Velocity
# ============================================================
fig2 = plt.figure(figsize=(16, 5))

# Linear Velocity
ax1 = plt.subplot(1, 3, 1)
ax1.plot(df['time'], df['vx'], label='Vx', linewidth=2)
ax1.plot(df['time'], df['vy'], label='Vy', linewidth=2)
ax1.plot(df['time'], df['vz'], label='Vz', linewidth=2)
ax1.set_ylabel('Linear Velocity (m/s)', fontsize=11)
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Linear Velocity (Body Frame)', fontsize=12, fontweight='bold')

# Angular Velocity
ax2 = plt.subplot(1, 3, 2)
ax2.plot(df['time'], df['wx'], label='Wx (roll rate)', linewidth=2)
ax2.plot(df['time'], df['wy'], label='Wy (pitch rate)', linewidth=2)
ax2.plot(df['time'], df['wz'], label='Wz (yaw rate)', linewidth=2)
ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=11)
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Angular Velocity (Body Frame)', fontsize=12, fontweight='bold')

# Velocity Magnitude
ax3 = plt.subplot(1, 3, 3)
vel_mag = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)
ax3.plot(df['time'], vel_mag, 'g-', linewidth=2, label='Speed')
ax3.set_ylabel('Speed (m/s)', fontsize=11)
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Total Speed (Velocity Magnitude)', fontsize=12, fontweight='bold')

plt.suptitle('NMPC - Velocity Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path2 = pic_dir / 'nmpc_velocity.png'
plt.savefig(pic_path2, dpi=150, bbox_inches='tight')
print(f"✓ Velocity plot saved: {pic_path2}")

# ============================================================
# FIGURE 3: Control Inputs
# ============================================================
fig3 = plt.figure(figsize=(14, 10))

# Motor Speeds - Individual
ax1 = plt.subplot(3, 2, 1)
ax1.plot(df['time'], df['u_front'], label='Front', linewidth=2)
ax1.set_ylabel('Front Motor (krpm)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Front Motor Speed', fontsize=12, fontweight='bold')

ax2 = plt.subplot(3, 2, 2)
ax2.plot(df['time'], df['u_left'], label='Left', linewidth=2, color='orange')
ax2.set_ylabel('Left Motor (krpm)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Left Motor Speed', fontsize=12, fontweight='bold')

ax3 = plt.subplot(3, 2, 3)
ax3.plot(df['time'], df['u_rear'], label='Rear', linewidth=2, color='green')
ax3.set_ylabel('Rear Motor (krpm)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Rear Motor Speed', fontsize=12, fontweight='bold')

ax4 = plt.subplot(3, 2, 4)
ax4.plot(df['time'], df['u_right'], label='Right', linewidth=2, color='red')
ax4.set_ylabel('Right Motor (krpm)', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_title('Right Motor Speed', fontsize=12, fontweight='bold')

# All motors together
ax5 = plt.subplot(3, 2, 5)
ax5.plot(df['time'], df['u_front'], label='Front', linewidth=1.5)
ax5.plot(df['time'], df['u_left'], label='Left', linewidth=1.5)
ax5.plot(df['time'], df['u_rear'], label='Rear', linewidth=1.5)
ax5.plot(df['time'], df['u_right'], label='Right', linewidth=1.5)
ax5.set_ylabel('Motor Speed (krpm)', fontsize=11)
ax5.set_xlabel('Time (s)', fontsize=11)
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_title('All Motors Combined', fontsize=12, fontweight='bold')

# Yaw Control
ax6 = plt.subplot(3, 2, 6)
ax6.plot(df['time'], df['u_yaw_bias'], 'g-', linewidth=2)
ax6.set_ylabel('Yaw Bias', fontsize=11)
ax6.set_xlabel('Time (s)', fontsize=11)
ax6.grid(True, alpha=0.3)
ax6.set_title('Yaw Control Bias', fontsize=12, fontweight='bold')

plt.suptitle('NMPC - Control Input Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path3 = pic_dir / 'nmpc_control.png'
plt.savefig(pic_path3, dpi=150, bbox_inches='tight')
print(f"✓ Control plot saved: {pic_path3}")
# ============================================================
# FIGURE 4: Attitude (Orientation)
# ============================================================
# Convert quaternions to Euler angles (roll, pitch, yaw)
def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.copysign(np.pi / 2, sinp),
                     np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw

roll, pitch, yaw = quat_to_euler(df['qw'].values, df['qx'].values, 
                                  df['qy'].values, df['qz'].values)

fig4 = plt.figure(figsize=(16, 5))

# Roll angle
ax1 = plt.subplot(1, 3, 1)
ax1.plot(df['time'], np.degrees(roll), 'b-', linewidth=2, label='Roll')
ax1.set_ylabel('Roll Angle (deg)', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('Roll Angle', fontsize=12, fontweight='bold')

# Pitch angle
ax2 = plt.subplot(1, 3, 2)
ax2.plot(df['time'], np.degrees(pitch), 'g-', linewidth=2, label='Pitch')
ax2.set_ylabel('Pitch Angle (deg)', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_title('Pitch Angle', fontsize=12, fontweight='bold')

# Yaw angle
ax3 = plt.subplot(1, 3, 3)
ax3.plot(df['time'], np.degrees(yaw), 'r-', linewidth=2, label='Yaw')
ax3.set_ylabel('Yaw Angle (deg)', fontsize=11)
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_title('Yaw Angle', fontsize=12, fontweight='bold')

# All Euler angles together
# ax4 = plt.subplot(2, 2, 4)
# ax4.plot(df['time'], np.degrees(roll), 'b-', linewidth=1.5, label='Roll')
# ax4.plot(df['time'], np.degrees(pitch), 'g-', linewidth=1.5, label='Pitch')
# ax4.plot(df['time'], np.degrees(yaw), 'r-', linewidth=1.5, label='Yaw')
# ax4.set_ylabel('Angle (deg)', fontsize=11)
# ax4.set_xlabel('Time (s)', fontsize=11)
# ax4.legend()
# ax4.grid(True, alpha=0.3)
# ax4.set_title('All Euler Angles', fontsize=12, fontweight='bold')

plt.suptitle('NMPC - Attitude Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path4 = pic_dir / 'nmpc_attitude.png'
plt.savefig(pic_path4, dpi=150, bbox_inches='tight')

print(f"✓ Attitude plot saved: {pic_path4}")

# ============================================================
# FIGURE 5: Performance Metrics
# ============================================================
fig5 = plt.figure(figsize=(14, 5))

# Position Error
ax1 = plt.subplot(1, 2, 1)
ax1.plot(df['time'], pos_error, 'r-', linewidth=2)
ax1.set_ylabel('Position Error (m)', fontsize=11)
ax1.set_xlabel('Time (s)', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_title('Position Tracking Error', fontsize=12, fontweight='bold')

# Solve Time
ax2 = plt.subplot(1, 2, 2)
ax2.plot(df['time'], df['solve_time']*1000, 'b-', linewidth=1.5)
ax2.set_ylabel('Solve Time (ms)', fontsize=11)
ax2.set_xlabel('Time (s)', fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_title('NMPC Solve Time', fontsize=12, fontweight='bold')
ax2.axhline(y=10, color='r', linestyle='--', linewidth=1, label='10ms (100Hz)')
ax2.legend()

plt.suptitle('NMPC - Performance Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
pic_path5 = pic_dir / 'nmpc_performance.png'
plt.savefig(pic_path5, dpi=150, bbox_inches='tight')
print(f"✓ Performance plot saved: {pic_path5}")

print("\n" + "="*50)
print("All plots generated successfully!")
print("="*50)

# Show all figures
plt.show()
