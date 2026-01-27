#!/usr/bin/env python3
# 控制性能分析脚本 - 调节时间、超调量、稳态误差分析
import pandas as pd

# Auto-detect log path based on script location
import os
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
log_path = project_root / "log" / "csv" / "nmpc_data.csv"

import matplotlib.pyplot as plt
import numpy as np

# Font configuration
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['font.monospace'] = ['DejaVu Sans Mono']
plt.rcParams['axes.unicode_minus'] = False

# ========== 性能指标计算函数 ==========

def calculate_settling_time(time, actual, goal, tolerance=0.02):
    """
    计算调节时间（Settling Time）
    
    参数:
        time: 时间数组
        actual: 实际值数组
        goal: 目标值数组
        tolerance: 容差范围（默认±2%）
    
    返回:
        settling_time: 调节时间（秒），如果未稳定则返回None
        settling_idx: 稳定时刻的索引
    """
    error = np.abs(actual - goal)
    goal_range = np.abs(goal)
    
    # 动态容差：目标值的±2%，但至少0.01m
    tolerance_threshold = np.maximum(goal_range * tolerance, 0.01)
    
    # 从后往前找，找到第一个超出容差的点
    settled_mask = error <= tolerance_threshold
    
    # 找到最后一次离开稳定区间的时刻
    for i in range(len(settled_mask) - 1, -1, -1):
        if not settled_mask[i]:
            if i + 1 < len(time):
                return time[i + 1], i + 1
            else:
                return None, None
    
    # 如果一直都在容差范围内，调节时间为0
    return time[0], 0


def calculate_overshoot(actual, goal):
    """
    计算超调量（Overshoot）
    
    参数:
        actual: 实际值数组
        goal: 目标值数组（假设为常值或缓变）
    
    返回:
        overshoot: 超调量（%）
        max_value: 最大值
        max_idx: 最大值索引
    """
    # 取目标值的平均（假设稳态目标）
    steady_goal = np.mean(goal[-100:])  # 最后100个点的平均
    
    # 如果是爬升过程，使用最终目标
    if np.abs(goal[-1] - goal[0]) > 0.1:
        steady_goal = goal[-1]
    
    max_value = np.max(actual)
    max_idx = np.argmax(actual)
    
    if steady_goal != 0:
        overshoot = ((max_value - steady_goal) / np.abs(steady_goal)) * 100
    else:
        overshoot = 0
    
    return overshoot, max_value, max_idx


def calculate_steady_state_error(actual, goal, start_idx=None):
    """
    计算稳态误差（Steady-State Error）
    
    参数:
        actual: 实际值数组
        goal: 目标值数组
        start_idx: 稳态开始的索引（如果为None，使用后20%数据）
    
    返回:
        mean_error: 平均稳态误差
        std_error: 稳态误差标准差
        max_error: 最大稳态误差
    """
    if start_idx is None:
        start_idx = int(len(actual) * 0.8)  # 使用后20%数据
    
    steady_actual = actual[start_idx:]
    steady_goal = goal[start_idx:]
    
    error = steady_actual - steady_goal
    
    mean_error = np.mean(error)
    std_error = np.std(error)
    max_error = np.max(np.abs(error))
    
    return mean_error, std_error, max_error


# ========== 主程序 ==========

# 读取数据
df = pd.read_csv(log_path)

# 创建图形窗口
fig = plt.figure(figsize=(20, 12))

# ========== Z轴位置跟踪与性能指标 ==========
ax1 = plt.subplot(2, 3, 1)

# 绘制Z轴位置
ax1.plot(df['time'], df['pz'], 'b-', label='Actual Z', linewidth=2)
ax1.plot(df['time'], df['goal_z'], 'r--', label='Goal Z', linewidth=1.5, alpha=0.7)

# 计算性能指标
settling_time_z, settling_idx_z = calculate_settling_time(
    df['time'].values, df['pz'].values, df['goal_z'].values, tolerance=0.02
)
overshoot_z, max_z, max_idx_z = calculate_overshoot(df['pz'].values, df['goal_z'].values)
mean_err_z, std_err_z, max_err_z = calculate_steady_state_error(
    df['pz'].values, df['goal_z'].values, start_idx=settling_idx_z
)

# 标注调节时间
if settling_time_z is not None:
    ax1.axvline(x=settling_time_z, color='g', linestyle=':', linewidth=1.5, 
                label=f'Settling Time: {settling_time_z:.2f}s')
    ax1.axhspan(df['goal_z'].iloc[-1] * 0.98, df['goal_z'].iloc[-1] * 1.02, 
                alpha=0.2, color='green', label='±2% Band')

# 标注超调
if overshoot_z > 0.5:  # 只在有明显超调时标注
    ax1.plot(df['time'].iloc[max_idx_z], max_z, 'ro', markersize=8)
    ax1.annotate(f'Max: {max_z:.3f}m\nOvershoot: {overshoot_z:.1f}%', 
                xy=(df['time'].iloc[max_idx_z], max_z),
                xytext=(df['time'].iloc[max_idx_z] + 2, max_z + 0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red')

ax1.set_xlabel('Time (s)', fontsize=11)
ax1.set_ylabel('Z Position (m)', fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Z-Axis Position Tracking', fontsize=12, fontweight='bold')

# ========== X轴位置跟踪 ==========
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df['time'], df['px'], 'b-', label='Actual X', linewidth=2)
ax2.plot(df['time'], df['goal_x'], 'r--', label='Goal X', linewidth=1.5, alpha=0.7)

settling_time_x, settling_idx_x = calculate_settling_time(
    df['time'].values, df['px'].values, df['goal_x'].values, tolerance=0.02
)
overshoot_x, max_x, max_idx_x = calculate_overshoot(df['px'].values, df['goal_x'].values)

if settling_time_x is not None:
    ax2.axvline(x=settling_time_x, color='g', linestyle=':', linewidth=1.5,
                label=f'Settling: {settling_time_x:.2f}s')

ax2.set_xlabel('Time (s)', fontsize=11)
ax2.set_ylabel('X Position (m)', fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title('X-Axis Position Tracking', fontsize=12, fontweight='bold')

# ========== Y轴位置跟踪 ==========
ax3 = plt.subplot(2, 3, 3)
ax3.plot(df['time'], df['py'], 'b-', label='Actual Y', linewidth=2)
ax3.plot(df['time'], df['goal_y'], 'r--', label='Goal Y', linewidth=1.5, alpha=0.7)

settling_time_y, settling_idx_y = calculate_settling_time(
    df['time'].values, df['py'].values, df['goal_y'].values, tolerance=0.02
)
overshoot_y, max_y, max_idx_y = calculate_overshoot(df['py'].values, df['goal_y'].values)

if settling_time_y is not None:
    ax3.axvline(x=settling_time_y, color='g', linestyle=':', linewidth=1.5,
                label=f'Settling: {settling_time_y:.2f}s')

ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('Y Position (m)', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_title('Y-Axis Position Tracking', fontsize=12, fontweight='bold')

# ========== 位置误差时间历史 ==========
ax4 = plt.subplot(2, 3, 4)

error_x = df['px'] - df['goal_x']
error_y = df['py'] - df['goal_y']
error_z = df['pz'] - df['goal_z']

ax4.plot(df['time'], error_x, label='X Error', linewidth=1.5, alpha=0.8)
ax4.plot(df['time'], error_y, label='Y Error', linewidth=1.5, alpha=0.8)
ax4.plot(df['time'], error_z, label='Z Error', linewidth=1.5, alpha=0.8)
ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

# 标注±2%容差带（基于Z轴目标）
if df['goal_z'].iloc[-1] != 0:
    tolerance_z = df['goal_z'].iloc[-1] * 0.02
    ax4.axhspan(-tolerance_z, tolerance_z, alpha=0.15, color='green', label='±2% Band (Z)')

ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Position Error (m)', fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_title('Position Tracking Error', fontsize=12, fontweight='bold')

# ========== 稳态误差统计 ==========
ax5 = plt.subplot(2, 3, 5)

# 计算三个轴的稳态误差
steady_start = max(settling_idx_x or 0, settling_idx_y or 0, settling_idx_z or 0)
if steady_start == 0:
    steady_start = int(len(df) * 0.8)

mean_err_x, std_err_x, max_err_x = calculate_steady_state_error(
    df['px'].values, df['goal_x'].values, start_idx=steady_start
)
mean_err_y, std_err_y, max_err_y = calculate_steady_state_error(
    df['py'].values, df['goal_y'].values, start_idx=steady_start
)

# 绘制误差箱线图
positions = [1, 2, 3]
error_data = [
    error_x.iloc[steady_start:],
    error_y.iloc[steady_start:],
    error_z.iloc[steady_start:]
]

bp = ax5.boxplot(error_data, positions=positions, widths=0.6, patch_artist=True,
                 tick_labels=['X', 'Y', 'Z'],
                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                 medianprops=dict(color='red', linewidth=2),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5))

ax5.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_ylabel('Steady-State Error (m)', fontsize=11)
ax5.set_xlabel('Axis', fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_title('Steady-State Error Distribution', fontsize=12, fontweight='bold')

# 添加数值标注
for i, (mean, std) in enumerate([(mean_err_x, std_err_x), 
                                   (mean_err_y, std_err_y), 
                                   (mean_err_z, std_err_z)]):
    ax5.text(i + 1, ax5.get_ylim()[1] * 0.9, 
            f'μ={mean:.4f}\nσ={std:.4f}',
            ha='center', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== 性能指标汇总 ==========
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# 计算总位置误差
total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
mean_total_err = np.mean(total_error.iloc[steady_start:])
max_total_err = np.max(total_error)

# 处理None值（如果从未稳定，显示为'N/A'）
st_x_str = f"{settling_time_x:.3f}" if settling_time_x is not None else "N/A"
st_y_str = f"{settling_time_y:.3f}" if settling_time_y is not None else "N/A"
st_z_str = f"{settling_time_z:.3f}" if settling_time_z is not None else "N/A"

performance_text = f"""
{'='*50}
      Performance Metrics Summary
{'='*50}

Settling Time (±2%):
  X: {st_x_str:>8} s    (error < 2% after {st_x_str}s)
  Y: {st_y_str:>8} s    (error < 2% after {st_y_str}s)
  Z: {st_z_str:>8} s    (error < 2% after {st_z_str}s)

Overshoot:
  X: {overshoot_x:6.2f}%    (max: {max_x:.4f} m)
  Y: {overshoot_y:6.2f}%    (max: {max_y:.4f} m)
  Z: {overshoot_z:6.2f}%    (max: {max_z:.4f} m)

Steady-State Error:
  X: {mean_err_x:7.4f} ± {std_err_x:.4f} m (max: {max_err_x:.4f} m)
  Y: {mean_err_y:7.4f} ± {std_err_y:.4f} m (max: {max_err_y:.4f} m)
  Z: {mean_err_z:7.4f} ± {std_err_z:.4f} m (max: {max_err_z:.4f} m)

Total Position Error:
  Mean: {mean_total_err:.4f} m
  Max:  {max_total_err:.4f} m
  RMS:  {np.sqrt(np.mean(total_error**2)):.4f} m

Data Statistics:
  Duration:      {df['time'].iloc[-1]:.2f} s
  Data points:   {len(df)}
  Sample rate:   {len(df)/df['time'].iloc[-1]:.1f} Hz
  Steady start:  {df['time'].iloc[steady_start]:.2f} s (point #{steady_start})

Solver Performance:
  Mean solve time: {df['solve_time'].mean()*1000:.2f} ms
  Max solve time:  {df['solve_time'].max()*1000:.2f} ms
  Min solve time:  {df['solve_time'].min()*1000:.2f} ms
{'='*50}
"""

ax6.text(0.05, 0.95, performance_text, 
         fontsize=9, 
         verticalalignment='top',
         family='sans-serif', 
         fontproperties={'weight': 'normal'},
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# ========== Adjust layout and save ==========
plt.suptitle('NMPC Control Performance Analysis - Metrics Summary', 
             fontsize=14, fontweight='bold', y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.985], h_pad=1.5, w_pad=2.0)

# 保存图片到log/pic目录
from pathlib import Path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
pic_dir = project_root / 'log' / 'pic'
pic_dir.mkdir(parents=True, exist_ok=True)
pic_path = pic_dir / 'control_performance_analysis.png'
plt.savefig(pic_path, dpi=150, bbox_inches='tight')
print(f'✓ 图片已保存: {pic_path}')
print("="*60)
print("Performance Analysis Complete")
print("="*60)
print(f"\nSettling Time (±2%):")
print(f"  X: {st_x_str}s | Y: {st_y_str}s | Z: {st_z_str}s")
print(f"\nOvershoot:")
print(f"  X: {overshoot_x:.2f}% | Y: {overshoot_y:.2f}% | Z: {overshoot_z:.2f}%")
print(f"\nSteady-State Error:")
print(f"  X: {mean_err_x:.4f}±{std_err_x:.4f}m")
print(f"  Y: {mean_err_y:.4f}±{std_err_y:.4f}m")
print(f"  Z: {mean_err_z:.4f}±{std_err_z:.4f}m")
print(f"\nTotal Position Error:")
print(f"  Mean: {mean_total_err:.4f}m | Max: {max_total_err:.4f}m")
print(f"\nPlot saved: control_performance_analysis.png")
print("="*60)

# 显示图片
plt.show()
