#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyboard Control Data Logging and Real-time Plotting Module
"""

import csv
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


class KeyboardControlLogger:
    """Keyboard control data logger"""
    
    def __init__(self, log_file='log/keyboard_control_log.csv'):
        """
        Initialize data logger
        
        Args:
            log_file: CSV log file path
        """
        self.log_file = log_file
        self.csv_file = open(log_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV header
        self.csv_writer.writerow([
            'time', 'mode', 'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_yaw_rate',
            'pos_x', 'pos_y', 'pos_z',
            'roll', 'pitch', 'yaw',
            'vel_x', 'vel_y', 'vel_z',
            'target_yaw'
        ])
        
    def log_data(self, time, mode, vel_cmd, pos, euler, vel, target_yaw):
        """
        Log one data entry
        
        Args:
            time: timestamp
            mode: control mode
            vel_cmd: velocity command [vx, vy, vz, yaw_rate]
            pos: position [x, y, z]
            euler: euler angles [roll, pitch, yaw] (radians)
            vel: velocity [vx, vy, vz]
            target_yaw: target yaw angle
        """
        self.csv_writer.writerow([
            time,
            mode,
            vel_cmd[0], vel_cmd[1], vel_cmd[2], vel_cmd[3],
            pos[0], pos[1], pos[2],
            euler[0], euler[1], euler[2],
            vel[0], vel[1], vel[2],
            target_yaw
        ])
        
    def close(self):
        """Close CSV file"""
        self.csv_file.close()


class KeyboardControlPlotter:
    """Keyboard control real-time plotter"""
    
    def __init__(self, buffer_size=500):
        """
        Initialize plotter
        
        Args:
            buffer_size: data buffer size (number of recent data points to display)
        """
        self.buffer_size = buffer_size
        
        # Initialize data buffer
        self.plot_data = {
            'time': deque(maxlen=buffer_size),
            'cmd_vx': deque(maxlen=buffer_size),
            'vel_x': deque(maxlen=buffer_size),
            'pos_x': deque(maxlen=buffer_size),
            'pos_y': deque(maxlen=buffer_size),
            'yaw': deque(maxlen=buffer_size),
            'target_yaw': deque(maxlen=buffer_size)
        }
        
        # Create figure
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Keyboard Control Real-time Monitor', fontsize=14)
        
        # Configure subplots
        self.axes[0, 0].set_title('Velocity Tracking (X-axis)')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].set_ylabel('Velocity (m/s)')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Position')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].set_ylabel('Position (m)')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Yaw Tracking')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].set_ylabel('Yaw (rad)')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('XY Trajectory')
        self.axes[1, 1].set_xlabel('X (m)')
        self.axes[1, 1].set_ylabel('Y (m)')
        self.axes[1, 1].grid(True)
        self.axes[1, 1].axis('equal')
        
        plt.tight_layout()
        
    def update_data(self, time, vel_cmd, pos, vel, yaw, target_yaw):
        """
        Update data buffer
        
        Args:
            time: timestamp
            vel_cmd: velocity command [vx, vy, vz, yaw_rate]
            pos: position [x, y, z]
            vel: velocity [vx, vy, vz]
            yaw: current yaw
            target_yaw: target yaw
        """
        self.plot_data['time'].append(time)
        self.plot_data['cmd_vx'].append(vel_cmd[0])
        self.plot_data['vel_x'].append(vel[0])
        self.plot_data['pos_x'].append(pos[0])
        self.plot_data['pos_y'].append(pos[1])
        self.plot_data['yaw'].append(yaw)
        self.plot_data['target_yaw'].append(target_yaw)
        
    def update_plot(self):
        """Update plot"""
        if len(self.plot_data['time']) < 2:
            return
            
        time = np.array(self.plot_data['time'])
        
        # Clear and redraw velocity tracking
        self.axes[0, 0].clear()
        self.axes[0, 0].plot(time, self.plot_data['cmd_vx'], 'r--', label='Command')
        self.axes[0, 0].plot(time, self.plot_data['vel_x'], 'b-', label='Actual')
        self.axes[0, 0].set_title('Velocity Tracking (X-axis)')
        self.axes[0, 0].set_xlabel('Time (s)')
        self.axes[0, 0].set_ylabel('Velocity (m/s)')
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(True)
        
        # Clear and redraw position
        self.axes[0, 1].clear()
        self.axes[0, 1].plot(time, self.plot_data['pos_x'], label='X')
        self.axes[0, 1].plot(time, self.plot_data['pos_y'], label='Y')
        self.axes[0, 1].set_title('Position')
        self.axes[0, 1].set_xlabel('Time (s)')
        self.axes[0, 1].set_ylabel('Position (m)')
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(True)
        
        # Clear and redraw yaw
        self.axes[1, 0].clear()
        self.axes[1, 0].plot(time, self.plot_data['target_yaw'], 'r--', label='Target')
        self.axes[1, 0].plot(time, self.plot_data['yaw'], 'b-', label='Actual')
        self.axes[1, 0].set_title('Yaw Tracking')
        self.axes[1, 0].set_xlabel('Time (s)')
        self.axes[1, 0].set_ylabel('Yaw (rad)')
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(True)
        
        # Clear and redraw XY trajectory
        self.axes[1, 1].clear()
        self.axes[1, 1].plot(self.plot_data['pos_x'], self.plot_data['pos_y'], 'b-')
        self.axes[1, 1].scatter(self.plot_data['pos_x'][-1], self.plot_data['pos_y'][-1], 
                               c='r', s=50, label='Current')
        self.axes[1, 1].set_title('XY Trajectory')
        self.axes[1, 1].set_xlabel('X (m)')
        self.axes[1, 1].set_ylabel('Y (m)')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True)
        self.axes[1, 1].axis('equal')
        
        plt.pause(0.001)
        
    def close(self):
        """Close plot window"""
        plt.ioff()
        plt.show()
