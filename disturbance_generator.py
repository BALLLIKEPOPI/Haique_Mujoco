#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扰动生成器 - 用于测试NMPC控制器的鲁棒性
支持脉冲、阶跃、斜波、正弦等多种扰动类型
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class DisturbanceGenerator:
    """扰动生成器类，支持多种扰动类型"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化扰动生成器
        
        Args:
            seed: 随机种子，用于可重复性测试
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_time = 0.0
        self.disturbances = []  # 扰动事件列表
        
    def add_pulse_disturbance(
        self,
        start_time: float,
        duration: float,
        force: np.ndarray = None,
        torque: np.ndarray = None
    ):
        """
        添加脉冲扰动
        
        Args:
            start_time: 开始时间（秒）
            duration: 持续时间（秒）
            force: 力扰动 [fx, fy, fz] (N)
            torque: 力矩扰动 [mx, my, mz] (Nm)
        """
        if force is None:
            force = np.zeros(3)
        if torque is None:
            torque = np.zeros(3)
            
        self.disturbances.append({
            'type': 'pulse',
            'start_time': start_time,
            'end_time': start_time + duration,
            'force': np.array(force, dtype=float),
            'torque': np.array(torque, dtype=float)
        })
        
    def add_step_disturbance(
        self,
        start_time: float,
        force: np.ndarray = None,
        torque: np.ndarray = None
    ):
        """
        添加阶跃扰动（持续到仿真结束）
        
        Args:
            start_time: 开始时间（秒）
            force: 力扰动 [fx, fy, fz] (N)
            torque: 力矩扰动 [mx, my, mz] (Nm)
        """
        if force is None:
            force = np.zeros(3)
        if torque is None:
            torque = np.zeros(3)
            
        self.disturbances.append({
            'type': 'step',
            'start_time': start_time,
            'end_time': float('inf'),
            'force': np.array(force, dtype=float),
            'torque': np.array(torque, dtype=float)
        })
        
    def add_ramp_disturbance(
        self,
        start_time: float,
        duration: float,
        final_force: np.ndarray = None,
        final_torque: np.ndarray = None
    ):
        """
        添加斜波扰动（线性增长）
        
        Args:
            start_time: 开始时间（秒）
            duration: 增长持续时间（秒）
            final_force: 最终力扰动 [fx, fy, fz] (N)
            final_torque: 最终力矩扰动 [mx, my, mz] (Nm)
        """
        if final_force is None:
            final_force = np.zeros(3)
        if final_torque is None:
            final_torque = np.zeros(3)
            
        self.disturbances.append({
            'type': 'ramp',
            'start_time': start_time,
            'end_time': start_time + duration,
            'final_force': np.array(final_force, dtype=float),
            'final_torque': np.array(final_torque, dtype=float)
        })
        
    def add_sine_disturbance(
        self,
        start_time: float,
        duration: float,
        amplitude_force: np.ndarray = None,
        amplitude_torque: np.ndarray = None,
        frequency: float = 1.0,
        phase: float = 0.0
    ):
        """
        添加正弦扰动
        
        Args:
            start_time: 开始时间（秒）
            duration: 持续时间（秒），None表示持续到仿真结束
            amplitude_force: 力振幅 [fx, fy, fz] (N)
            amplitude_torque: 力矩振幅 [mx, my, mz] (Nm)
            frequency: 频率 (Hz)
            phase: 初始相位 (弧度)
        """
        if amplitude_force is None:
            amplitude_force = np.zeros(3)
        if amplitude_torque is None:
            amplitude_torque = np.zeros(3)
            
        self.disturbances.append({
            'type': 'sine',
            'start_time': start_time,
            'end_time': start_time + duration if duration is not None else float('inf'),
            'amplitude_force': np.array(amplitude_force, dtype=float),
            'amplitude_torque': np.array(amplitude_torque, dtype=float),
            'frequency': frequency,
            'phase': phase
        })
        
    def add_random_disturbances(
        self,
        start_time: float,
        end_time: float,
        num_events: int = 5,
        max_force: float = 20.0,
        max_torque: float = 5.0
    ):
        """
        在指定时间范围内随机添加多个扰动事件
        
        Args:
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            num_events: 扰动事件数量
            max_force: 最大力扰动幅值 (N)
            max_torque: 最大力矩扰动幅值 (Nm)
        """
        disturbance_types = ['pulse', 'step', 'sine']
        
        for _ in range(num_events):
            event_time = np.random.uniform(start_time, end_time)
            dist_type = np.random.choice(disturbance_types)
            
            # 随机生成力和力矩
            force = np.random.uniform(-max_force, max_force, 3)
            torque = np.random.uniform(-max_torque, max_torque, 3)
            
            if dist_type == 'pulse':
                duration = np.random.uniform(0.5, 2.0)
                self.add_pulse_disturbance(event_time, duration, force, torque)
            elif dist_type == 'step':
                # 阶跃扰动幅值减半，避免太强
                self.add_step_disturbance(event_time, force * 0.5, torque * 0.5)
            elif dist_type == 'sine':
                duration = np.random.uniform(3.0, 8.0)
                frequency = np.random.uniform(0.5, 2.0)
                self.add_sine_disturbance(
                    event_time, duration, force * 0.7, torque * 0.7, frequency
                )
                
    def get_disturbance(self, time: float) -> Dict[str, np.ndarray]:
        """
        获取当前时刻的总扰动
        
        Args:
            time: 当前时间（秒）
            
        Returns:
            包含'force'和'torque'的字典
        """
        self.current_time = time
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        for dist in self.disturbances:
            if dist['start_time'] <= time < dist['end_time']:
                if dist['type'] == 'pulse':
                    total_force += dist['force']
                    total_torque += dist['torque']
                    
                elif dist['type'] == 'step':
                    total_force += dist['force']
                    total_torque += dist['torque']
                    
                elif dist['type'] == 'ramp':
                    # 计算斜波进度 [0, 1]
                    progress = (time - dist['start_time']) / (dist['end_time'] - dist['start_time'])
                    progress = np.clip(progress, 0.0, 1.0)
                    total_force += progress * dist['final_force']
                    total_torque += progress * dist['final_torque']
                    
                elif dist['type'] == 'sine':
                    # 计算正弦扰动
                    t_rel = time - dist['start_time']
                    omega = 2 * np.pi * dist['frequency']
                    value = np.sin(omega * t_rel + dist['phase'])
                    total_force += value * dist['amplitude_force']
                    total_torque += value * dist['amplitude_torque']
        
        return {
            'force': total_force,
            'torque': total_torque
        }
    
    def clear_disturbances(self):
        """清除所有扰动事件"""
        self.disturbances = []
        
    def print_summary(self):
        """打印扰动事件摘要"""
        print("\n" + "="*60)
        print("扰动事件摘要")
        print("="*60)
        print(f"总扰动事件数: {len(self.disturbances)}")
        
        type_counts = {}
        for dist in self.disturbances:
            dtype = dist['type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        print("\n扰动类型统计:")
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count} 个")
        
        print("\n扰动事件详情:")
        for i, dist in enumerate(sorted(self.disturbances, key=lambda x: x['start_time'])):
            print(f"\n事件 {i+1}: {dist['type'].upper()}")
            print(f"  时间: {dist['start_time']:.2f}s - {dist['end_time']:.2f}s")
            
            if dist['type'] in ['pulse', 'step']:
                print(f"  力: {dist['force']}")
                print(f"  力矩: {dist['torque']}")
            elif dist['type'] == 'ramp':
                print(f"  最终力: {dist['final_force']}")
                print(f"  最终力矩: {dist['final_torque']}")
            elif dist['type'] == 'sine':
                print(f"  力振幅: {dist['amplitude_force']}")
                print(f"  力矩振幅: {dist['amplitude_torque']}")
                print(f"  频率: {dist['frequency']:.2f} Hz")
        
        print("\n" + "="*60 + "\n")


# 预定义的测试场景
class DisturbanceScenarios:
    """预定义的扰动测试场景"""
    
    @staticmethod
    def mild_disturbance(generator: DisturbanceGenerator):
        """温和扰动场景 - 用于基础测试"""
        # 5秒后施加一个短脉冲
        generator.add_pulse_disturbance(
            start_time=5.0,
            duration=0.5,
            force=[3.0, 0.0, 0.0],
            torque=[0.0, 0.0, 2.0]
        )
        
        # 10秒后施加一个正弦扰动
        generator.add_sine_disturbance(
            start_time=10.0,
            duration=5.0,
            amplitude_force=[5.0, 5.0, 0.0],
            frequency=1.0
        )
        
    @staticmethod
    def moderate_disturbance(generator: DisturbanceGenerator):
        """中等扰动场景 - 用于常规测试"""
        # 3秒后施加阶跃力扰动
        generator.add_step_disturbance(
            start_time=3.0,
            force=[8.0, -5.0, 0.0]
        )
        
        # 8秒后施加脉冲力矩
        generator.add_pulse_disturbance(
            start_time=8.0,
            duration=1.0,
            torque=[3.0, 3.0, 0.0]
        )
        
        # 15秒后施加斜波扰动
        generator.add_ramp_disturbance(
            start_time=15.0,
            duration=3.0,
            final_force=[0.0, 0.0, 15.0]
        )
        
    @staticmethod
    def severe_disturbance(generator: DisturbanceGenerator):
        """严重扰动场景 - 用于极限测试"""
        # 2秒后施加强力脉冲
        generator.add_pulse_disturbance(
            start_time=2.0,
            duration=0.8,
            force=[25.0, 15.0, 10.0],
            torque=[5.0, 5.0, 3.0]
        )
        
        # 6秒后施加持续阶跃
        generator.add_step_disturbance(
            start_time=6.0,
            force=[15.0, -10.0, 5.0],
            torque=[2.0, -2.0, 1.0]
        )
        
        # 12秒后施加高频正弦扰动
        generator.add_sine_disturbance(
            start_time=12.0,
            duration=8.0,
            amplitude_force=[12.0, 12.0, 8.0],
            amplitude_torque=[4.0, 4.0, 2.0],
            frequency=2.0
        )
        
    @staticmethod
    def random_disturbance(generator: DisturbanceGenerator, severity: str = 'moderate'):
        """随机扰动场景"""
        if severity == 'mild':
            max_force, max_torque, num_events = 10.0, 3.0, 3
        elif severity == 'moderate':
            max_force, max_torque, num_events = 15.0, 5.0, 5
        elif severity == 'severe':
            max_force, max_torque, num_events = 25.0, 8.0, 8
        else:
            max_force, max_torque, num_events = 15.0, 5.0, 5
            
        generator.add_random_disturbances(
            start_time=3.0,
            end_time=25.0,
            num_events=num_events,
            max_force=max_force,
            max_torque=max_torque
        )


# 测试代码
if __name__ == "__main__":
    # 创建生成器
    gen = DisturbanceGenerator(seed=42)
    
    # 使用预定义场景
    print("测试中等扰动场景:")
    DisturbanceScenarios.moderate_disturbance(gen)
    gen.print_summary()
    
    # 测试获取扰动
    print("\n时间序列测试:")
    for t in [0.0, 3.5, 8.5, 16.0, 20.0]:
        dist = gen.get_disturbance(t)
        print(f"t={t:.1f}s: F={dist['force']}, M={dist['torque']}")
