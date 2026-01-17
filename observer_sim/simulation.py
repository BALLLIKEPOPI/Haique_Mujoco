#!/usr/bin/env python3
"""
ESO观测器数值仿真
从CSV读取控制量，对比标称模型、扰动模型和ESO估计
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from forward_dynamics import ForwardDynamics
from eso_observer import ESO_Observer


class SimulationRunner:
    """仿真执行器"""
    
    def __init__(self, csv_path, config_path=None):
        # 读取CSV数据
        self.df = pd.read_csv(csv_path)
        print(f"✓ 加载CSV数据: {len(self.df)} 条记录")
        
        # 初始化动力学模型
        self.dynamics = ForwardDynamics(
            g0=9.8066,
            mass=4.672,
            inertia=(0.10170715, 0.10222875, 0.16095642),
            Ct=0.0267,
            Cd=0.00111,
            dq=0.605,
            alpha_offset=-0.1,
            beta_offset=0.0,
            k_yaw_lr=1.5,
            k_yaw_other=0.3,
        )
        
        # 初始化ESO观测器
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        
        self.eso = ESO_Observer(
            dt=0.01,
            mode='underwater',
            left_servo_offset=-0.1,
            right_servo_offset=0.0,
            config_path=config_path
        )
        
        print("✓ 初始化动力学模型和ESO观测器")
    
    def run(self, disturbance_profile=None):
        """
        运行仿真
        
        Args:
            disturbance_profile: 扰动配置 {'force': callable(t), 'torque': callable(t)}
                                 如果为None，则从CSV读取actual_dist_*
        
        Returns:
            results: dict包含所有仿真结果
        """
        # 初始状态
        state_nominal = np.array([
            self.df.iloc[0]['px'],
            self.df.iloc[0]['py'],
            self.df.iloc[0]['pz'],
            self.df.iloc[0]['qw'],
            self.df.iloc[0]['qx'],
            self.df.iloc[0]['qy'],
            self.df.iloc[0]['qz'],
            self.df.iloc[0]['vx'],
            self.df.iloc[0]['vy'],
            self.df.iloc[0]['vz'],
            self.df.iloc[0]['wx'],
            self.df.iloc[0]['wy'],
            self.df.iloc[0]['wz'],
        ])
        state_disturbed = state_nominal.copy()
        
        # 存储结果
        results = {
            'time': [],
            'state_nominal': [],
            'state_disturbed': [],
            'eso_force': [],
            'eso_torque': [],
            'actual_force': [],
            'actual_torque': [],
        }
        
        # 检查CSV是否有实际扰动数据
        has_actual_dist = all(col in self.df.columns for col in 
                             ['actual_dist_fx', 'actual_dist_fy', 'actual_dist_fz',
                              'actual_dist_mx', 'actual_dist_my', 'actual_dist_mz'])
        
        print(f"开始仿真... (扰动模式: {'CSV实际扰动' if has_actual_dist else '自定义扰动'})")
        
        # 逐步仿真
        for i in range(len(self.df) - 1):
            t = self.df.iloc[i]['time']
            
            # 获取控制量
            control = np.array([
                self.df.iloc[i]['u1'],
                self.df.iloc[i]['u2'],
                self.df.iloc[i]['u3'],
                self.df.iloc[i]['u4'],
                self.df.iloc[i]['u5'],
                self.df.iloc[i]['u6'],
                self.df.iloc[i]['u7'],
                self.df.iloc[i]['u8'],
                self.df.iloc[i]['u_alpha'],
                self.df.iloc[i]['u_beta'],
            ])
            
            # 获取扰动
            if has_actual_dist and disturbance_profile is None:
                disturbance = {
                    'force': np.array([
                        self.df.iloc[i]['actual_dist_fx'],
                        self.df.iloc[i]['actual_dist_fy'],
                        self.df.iloc[i]['actual_dist_fz']
                    ]),
                    'torque': np.array([
                        self.df.iloc[i]['actual_dist_mx'],
                        self.df.iloc[i]['actual_dist_my'],
                        self.df.iloc[i]['actual_dist_mz']
                    ])
                }
            elif disturbance_profile is not None:
                disturbance = {
                    'force': disturbance_profile['force'](t),
                    'torque': disturbance_profile['torque'](t)
                }
            else:
                disturbance = {'force': np.zeros(3), 'torque': np.zeros(3)}
            
            # 计算时间步长
            dt = self.df.iloc[i+1]['time'] - t if i < len(self.df)-1 else 0.01
            
            # 标称模型仿真（无扰动）
            state_nominal = self.dynamics.integrate_rk4(state_nominal, control, dt, None)
            
            # 扰动模型仿真（有扰动）
            state_disturbed = self.dynamics.integrate_rk4(state_disturbed, control, dt, disturbance)
            
            # ESO观测器更新
            state_obs = state_disturbed[7:13]  # [vx, vy, vz, wx, wy, wz]
            quat = state_disturbed[3:7]
            eso_force, eso_torque = self.eso.update(state_obs, control, quat)
            
            # 记录结果
            results['time'].append(t)
            results['state_nominal'].append(state_nominal.copy())
            results['state_disturbed'].append(state_disturbed.copy())
            results['eso_force'].append(eso_force.copy())
            results['eso_torque'].append(eso_torque.copy())
            results['actual_force'].append(disturbance['force'].copy())
            results['actual_torque'].append(disturbance['torque'].copy())
            
            if i % 100 == 0:
                print(f"  进度: {i}/{len(self.df)} ({100*i/len(self.df):.1f}%)")
        
        # 转换为numpy数组
        for key in results:
            if key != 'time':
                results[key] = np.array(results[key])
        
        print(f"✓ 仿真完成: {len(results['time'])} 个时间步")
        return results


def create_test_disturbance():
    """创建测试扰动配置"""
    def force_func(t):
        # 5秒后施加阶跃扰动（仅Fx，Fy和Fz设为0）
        if t > 5.0:
            return np.array([
            4.0,  # Fx 保留
            0.0,  # Fy = 0
            0.0   # Fz = 0
        ])
        else:
            return np.zeros(3)
    
    def torque_func(t):
        # 10秒后施加正弦扰动（仅Mz，Mx和My设为0）
        if t > 10.0:
            return np.array([
                0.0,  # Mx = 0
                0.0,  # My = 0,
                0.2 * np.sin(2 * np.pi * 0.2 * t)
            ])
        else:
            return np.zeros(3)
    
    return {'force': force_func, 'torque': torque_func}


if __name__ == '__main__':
    # 运行仿真
    csv_path = '../log/nmpc_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到CSV文件: {csv_path}")
        sys.exit(1)
    
    sim = SimulationRunner(csv_path)
    
    # 使用测试扰动（人工添加扰动来验证ESO观测器）
    print("\n【扰动配置】")
    print("  5-10s:  Fx = 3.0 N (阶跃)")
    print("  10-12s: Fy = 2.5 N (脉冲)")
    print("  15-20s: Fz = 2.0*sin(t) N (正弦)")
    print("  20s+:   组合扰动")
    print("  8-13s:  Mx = 0.5 Nm (阶跃)")
    print("  12-18s: My = 0.3*sin(t) Nm (正弦)")
    print("  18s+:   Mz 斜坡增长\n")
    
    disturbance = create_test_disturbance()
    results = sim.run(disturbance_profile=disturbance)
    
    # 保存结果
    output_path = 'simulation_results.npz'
    np.savez(output_path, **results)
    print(f"\n✓ 仿真结果已保存到: {output_path}")
    
    # 显示统计信息
    print("\n【仿真统计】")
    print(f"时间范围: {results['time'][0]:.2f}s ~ {results['time'][-1]:.2f}s")
    print(f"扰动力统计:")
    print(f"  实际: max={np.max(np.abs(results['actual_force'])):.3f} N")
    print(f"  ESO:  max={np.max(np.abs(results['eso_force'])):.3f} N")
    print(f"扰动力矩统计:")
    print(f"  实际: max={np.max(np.abs(results['actual_torque'])):.4f} Nm")
    print(f"  ESO:  max={np.max(np.abs(results['eso_torque'])):.4f} Nm")
