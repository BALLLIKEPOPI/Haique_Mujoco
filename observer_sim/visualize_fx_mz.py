#!/usr/bin/env python3
"""
Visualize Fx and Mz disturbance observation results
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_fx_mz_results(results, save_path='fx_mz_visualization.png'):
    """
    Visualize Fx and Mz disturbance comparison
    
    Args:
        results: simulation.py output results dict
        save_path: save path
    """
    time = np.array(results['time'])
    eso_force = results['eso_force']
    eso_torque = results['eso_torque']
    actual_force = results['actual_force']
    actual_torque = results['actual_torque']
    
    # 创建图形
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3)
    
    # ========== Fx 扰动力对比 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, actual_force[:, 0], 'r-', linewidth=2.5, label='Actual Fx', alpha=0.8)
    ax1.plot(time, eso_force[:, 0], 'b--', linewidth=1.8, label='ESO Fx', alpha=0.7)
    ax1.set_ylabel('Disturbance Force Fx (N)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Disturbance Force Fx: Actual vs ESO', fontsize=13, fontweight='bold')
    
    # 添加误差统计
    fx_actual_max = np.max(np.abs(actual_force[:, 0]))
    fx_eso_max = np.max(np.abs(eso_force[:, 0]))
    fx_error = (fx_eso_max / fx_actual_max - 1) * 100 if fx_actual_max > 0 else 0
    
    textstr = f'Actual Peak: {fx_actual_max:.3f} N\n'
    textstr += f'ESO Peak: {fx_eso_max:.3f} N\n'
    textstr += f'Error: {fx_error:+.1f}%'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== Mz 扰动力矩对比 ==========
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time, actual_torque[:, 2], 'r-', linewidth=2.5, label='Actual Mz', alpha=0.8)
    ax2.plot(time, eso_torque[:, 2], 'b--', linewidth=1.8, label='ESO Mz', alpha=0.7)
    ax2.set_ylabel('Disturbance Torque Mz (Nm)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Disturbance Torque Mz: Actual vs ESO', fontsize=13, fontweight='bold')
    
    # 添加误差统计
    mz_actual_max = np.max(np.abs(actual_torque[:, 2]))
    mz_eso_max = np.max(np.abs(eso_torque[:, 2]))
    mz_error = (mz_eso_max / mz_actual_max - 1) * 100 if mz_actual_max > 0 else 0
    
    textstr = f'Actual Peak: {mz_actual_max:.4f} Nm\n'
    textstr += f'ESO Peak: {mz_eso_max:.4f} Nm\n'
    textstr += f'Error: {mz_error:+.1f}%'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('ESO Disturbance Observer Performance - Fx & Mz', fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")
    
    # Show
    plt.show()


if __name__ == '__main__':
    import sys
    
    # Load simulation results
    results_path = 'simulation_results.npz'
    
    try:
        data = np.load(results_path)
        results = {key: data[key] for key in data.files}
        print(f"Loaded: {results_path}")
        
        # Visualize
        plot_fx_mz_results(results)
        
    except FileNotFoundError:
        print(f"Error: File not found: {results_path}")
        print("Please run simulation.py first")
        sys.exit(1)
