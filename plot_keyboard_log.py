#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline visualization script for keyboard control log data
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_keyboard_log(csv_file, save_fig=None, show_plot=True):
    """
    Plot keyboard control log data
    
    Args:
        csv_file: Path to CSV log file
        save_fig: Path to save figure (optional)
        show_plot: Whether to display the plot
    """
    # Read CSV data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    if len(df) == 0:
        print("Error: CSV file is empty")
        return
    
    print(f"Loaded {len(df)} data points from {csv_file}")
    
    # Calculate velocity norm
    df['v_norm'] = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
    df['cmd_v_norm'] = np.sqrt(df['cmd_vx']**2 + df['cmd_vy']**2)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Keyboard Control Log Analysis', fontsize=16, fontweight='bold')
    
    # Subplot 1: Velocity Norm Tracking
    axes[0, 0].plot(df['time'], df['cmd_v_norm'], 'r--', label='Command Norm', linewidth=1.5)
    axes[0, 0].plot(df['time'], df['v_norm'], 'b-', label='Actual Norm', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_title('Velocity Norm Tracking', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Velocity (m/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Position (X and Y)
    axes[0, 1].plot(df['time'], df['pos_x'], label='X', linewidth=1.5)
    axes[0, 1].plot(df['time'], df['pos_y'], label='Y', linewidth=1.5)
    axes[0, 1].plot(df['time'], df['pos_z'], label='Z', linewidth=1.5, alpha=0.6)
    axes[0, 1].set_title('Position', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Yaw Tracking
    axes[1, 0].plot(df['time'], df['target_yaw'], 'r--', label='Target', linewidth=1.5)
    axes[1, 0].plot(df['time'], df['yaw'], 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    axes[1, 0].set_title('Yaw Tracking', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Yaw (rad)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: XY Trajectory
    axes[1, 1].plot(df['pos_x'], df['pos_y'], 'b-', linewidth=1.5, alpha=0.7)
    axes[1, 1].scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], 
                      c='g', s=100, marker='o', label='Start', zorder=5)
    axes[1, 1].scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], 
                      c='r', s=100, marker='x', label='End', zorder=5)
    axes[1, 1].set_title('XY Trajectory', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_fig}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_statistics(csv_file):
    """Print statistics of the log data"""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    print("\n" + "="*70)
    print("Log Data Statistics")
    print("="*70)
    
    duration = df['time'].iloc[-1] - df['time'].iloc[0]
    print(f"\nDuration: {duration:.2f} seconds")
    print(f"Data points: {len(df)}")
    print(f"Average sample rate: {len(df)/duration:.1f} Hz")
    
    print(f"\nPosition Range:")
    print(f"  X: [{df['pos_x'].min():.2f}, {df['pos_x'].max():.2f}] m")
    print(f"  Y: [{df['pos_y'].min():.2f}, {df['pos_y'].max():.2f}] m")
    print(f"  Z: [{df['pos_z'].min():.2f}, {df['pos_z'].max():.2f}] m")
    
    print(f"\nVelocity Statistics:")
    print(f"  X: mean={df['vel_x'].mean():.2f}, max={df['vel_x'].abs().max():.2f} m/s")
    print(f"  Y: mean={df['vel_y'].mean():.2f}, max={df['vel_y'].abs().max():.2f} m/s")
    print(f"  Z: mean={df['vel_z'].mean():.2f}, max={df['vel_z'].abs().max():.2f} m/s")
    
    # Calculate velocity norm for statistics (XY plane only)
    v_norm = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
    cmd_v_norm = np.sqrt(df['cmd_vx']**2 + df['cmd_vy']**2)
    print(f"  Norm (XY): mean={v_norm.mean():.2f}, max={v_norm.max():.2f} m/s")
    print(f"  Cmd Norm (XY): mean={cmd_v_norm.mean():.2f}, max={cmd_v_norm.max():.2f} m/s")
    
    print(f"\nYaw Statistics:")
    print(f"  Range: [{np.degrees(df['yaw'].min()):.1f}, {np.degrees(df['yaw'].max()):.1f}] degrees")
    print(f"  Total rotation: {np.degrees(df['yaw'].iloc[-1] - df['yaw'].iloc[0]):.1f} degrees")
    
    # Mode statistics
    mode_counts = df['mode'].value_counts()
    print(f"\nControl Mode Distribution:")
    for mode, count in mode_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {mode}: {count} samples ({percentage:.1f}%)")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Offline visualization of keyboard control log data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot latest log file
  python3 plot_keyboard_log.py
  
  # Plot specific log file
  python3 plot_keyboard_log.py --file log/my_flight.csv
  
  # Save plot to image
  python3 plot_keyboard_log.py --save plots/analysis.png
  
  # Show statistics only (no plot)
  python3 plot_keyboard_log.py --stats-only
  
  # Save and don't show plot
  python3 plot_keyboard_log.py --save result.png --no-show
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default='log/keyboard_control_log.csv',
        help='Path to CSV log file (default: log/keyboard_control_log.csv)'
    )
    
    parser.add_argument(
        '--save', '-s',
        type=str,
        help='Save figure to file (e.g., plot.png, plot.pdf)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot window'
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, do not plot'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return
    
    # Print statistics
    print_statistics(args.file)
    
    # Plot data (unless stats-only)
    if not args.stats_only:
        plot_keyboard_log(
            args.file,
            save_fig=args.save,
            show_plot=not args.no_show
        )


if __name__ == '__main__':
    main()
