#!/usr/bin/env python3
"""
可视化仿真结果
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_simulation_results(results, save_path='simulation_visualization.png'):
    """
    可视化仿真结果
    
    Args:
        results: simulation.py输出的结果字典
        save_path: 保存路径
    """
    time = np.array(results['time'])
    state_nominal = results['state_nominal']
    state_disturbed = results['state_disturbed']
    eso_force = results['eso_force']
    eso_torque = results['eso_torque']
    actual_force = results['actual_force']
    actual_torque = results['actual_torque']
    
    # 创建图形
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== 第1行：位置对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, state_nominal[:, 0], 'b-', label='Nominal', linewidth=1.5)
    ax1.plot(time, state_disturbed[:, 0], 'r--', label='Disturbed', linewidth=1.5)
    ax1.set_ylabel('X Position (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Position Comparison: X')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, state_nominal[:, 1], 'b-', label='Nominal', linewidth=1.5)
    ax2.plot(time, state_disturbed[:, 1], 'r--', label='Disturbed', linewidth=1.5)
    ax2.set_ylabel('Y Position (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Position Comparison: Y')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time, state_nominal[:, 2], 'b-', label='Nominal', linewidth=1.5)
    ax3.plot(time, state_disturbed[:, 2], 'r--', label='Disturbed', linewidth=1.5)
    ax3.set_ylabel('Z Position (m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Position Comparison: Z')
    
    # ========== 第2行：速度对比 ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time, state_nominal[:, 7], 'b-', label='Nominal Vx', linewidth=1.2)
    ax4.plot(time, state_disturbed[:, 7], 'r--', label='Disturbed Vx', linewidth=1.2)
    ax4.set_ylabel('Velocity (m/s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Linear Velocity: Vx')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(time, state_nominal[:, 8], 'b-', label='Nominal Vy', linewidth=1.2)
    ax5.plot(time, state_disturbed[:, 8], 'r--', label='Disturbed Vy', linewidth=1.2)
    ax5.set_ylabel('Velocity (m/s)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Linear Velocity: Vy')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(time, state_nominal[:, 9], 'b-', label='Nominal Vz', linewidth=1.2)
    ax6.plot(time, state_disturbed[:, 9], 'r--', label='Disturbed Vz', linewidth=1.2)
    ax6.set_ylabel('Velocity (m/s)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Linear Velocity: Vz')
    
    # ========== 第3行：扰动力对比 ==========
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(time, actual_force[:, 0], 'r-', linewidth=2, label='Actual Fx', alpha=0.8)
    ax7.plot(time, eso_force[:, 0], 'b--', linewidth=1.5, label='ESO Fx', alpha=0.7)
    ax7.set_ylabel('Force (N)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Disturbance Force: Fx')
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(time, actual_force[:, 1], 'r-', linewidth=2, label='Actual Fy', alpha=0.8)
    ax8.plot(time, eso_force[:, 1], 'b--', linewidth=1.5, label='ESO Fy', alpha=0.7)
    ax8.set_ylabel('Force (N)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Disturbance Force: Fy')
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(time, actual_force[:, 2], 'r-', linewidth=2, label='Actual Fz', alpha=0.8)
    ax9.plot(time, eso_force[:, 2], 'b--', linewidth=1.5, label='ESO Fz', alpha=0.7)
    ax9.set_ylabel('Force (N)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_title('Disturbance Force: Fz')
    
    # ========== 第4行：扰动力矩对比 ==========
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(time, actual_torque[:, 0], 'r-', linewidth=2, label='Actual Mx', alpha=0.8)
    ax10.plot(time, eso_torque[:, 0], 'b--', linewidth=1.5, label='ESO Mx', alpha=0.7)
    ax10.set_ylabel('Torque (Nm)')
    ax10.set_xlabel('Time (s)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    ax10.set_title('Disturbance Torque: Mx')
    
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(time, actual_torque[:, 1], 'r-', linewidth=2, label='Actual My', alpha=0.8)
    ax11.plot(time, eso_torque[:, 1], 'b--', linewidth=1.5, label='ESO My', alpha=0.7)
    ax11.set_ylabel('Torque (Nm)')
    ax11.set_xlabel('Time (s)')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_title('Disturbance Torque: My')
    
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(time, actual_torque[:, 2], 'r-', linewidth=2, label='Actual Mz', alpha=0.8)
    ax12.plot(time, eso_torque[:, 2], 'b--', linewidth=1.5, label='ESO Mz', alpha=0.7)
    ax12.set_ylabel('Torque (Nm)')
    ax12.set_xlabel('Time (s)')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    ax12.set_title('Disturbance Torque: Mz')
    
    plt.suptitle('ESO Observer Simulation Results', fontsize=16, fontweight='bold')
    
    # 保存
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化已保存到: {save_path}")
    
    # 显示
    plt.show()


if __name__ == '__main__':
    import sys
    
    # 加载仿真结果
    results_path = 'simulation_results.npz'
    
    try:
        data = np.load(results_path)
        results = {key: data[key] for key in data.files}
        print(f"✓ 加载仿真结果: {results_path}")
        
        # 可视化
        plot_simulation_results(results)
        
    except FileNotFoundError:
        print(f"❌ 找不到仿真结果文件: {results_path}")
        print("请先运行 simulation.py 生成仿真结果")
        sys.exit(1)
