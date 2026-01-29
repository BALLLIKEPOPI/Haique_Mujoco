#!/usr/bin/env python3
"""
Kinodynamic RRT 规划器 (基于运动学约束的快速扩展随机树)
用于水下过驱动无人机 (8电机 + 2舵机)
"""

import numpy as np
import casadi as ca
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.export_model_underwater import export_model_underwater


class KinodynamicRRT:
    """基于动力学约束的 RRT 路径规划器"""
    
    def __init__(self, start_state, goal_pos, bounds, obstacle_list=None, config=None):
        """
        参数:
            start_state: 初始状态 [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            goal_pos: 目标位置 [x, y, z]
            bounds: 边界 [x_min, x_max, y_min, y_max, z_min, z_max]
            obstacle_list: 障碍物列表 (暂未实现)
            config: 配置参数字典
        """
        self.start = np.array(start_state)
        self.goal = np.array(goal_pos)
        self.bounds = bounds
        self.obstacles = obstacle_list or []
        
        # 配置参数
        default_config = {
            'max_iter': 1000,           # 最大迭代次数
            'dt_sim': 0.2,              # 动力学积分步长 (s)
            'goal_sample_rate': 0.1,    # 采样目标点的概率
            'goal_tolerance': 0.3,      # 目标容差 (m)
            'num_control_samples': 10,  # 每次扩展尝试的控制输入数量
            'w_pos': 1.0,               # 位置距离权重
            'w_vel': 0.1,               # 速度距离权重
            'max_thrust': 12.0,         # 单电机最大推力 (N)
            'servo_range': 0.5,         # 舵机角度范围 (rad)
        }
        self.config = {**default_config, **(config or {})}
        
        # 加载水下动力学模型
        print("加载动力学模型...")
        model = export_model_underwater()
        self.f_expl = model.f_expl_expr  # 显式动力学方程
        self.x_sym = model.x
        self.u_sym = model.u
        
        # 创建 RK4 积分器
        print(f"创建积分器 (dt={self.config['dt_sim']}s)...")
        self._create_integrator()
        
        # RRT 树结构: List of {'state', 'parent', 'cost', 'control'}
        self.tree = [{
            'state': self.start,
            'parent': None,
            'cost': 0.0,
            'control': None
        }]
    
    def _create_integrator(self):
        """创建 RK4 积分器用于动力学推演"""
        dt = self.config['dt_sim']
        
        # RK4 积分
        k1 = self.f_expl
        k2 = ca.substitute(self.f_expl, self.x_sym, self.x_sym + dt/2 * k1)
        k3 = ca.substitute(self.f_expl, self.x_sym, self.x_sym + dt/2 * k2)
        k4 = ca.substitute(self.f_expl, self.x_sym, self.x_sym + dt * k3)
        x_next = self.x_sym + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # 创建 CasADi 函数
        self.integrator = ca.Function('integrator', 
                                       [self.x_sym, self.u_sym], 
                                       [x_next],
                                       ['x0', 'u'], ['xf'])
    
    def get_random_control(self):
        """生成随机控制输入"""
        # u: [f1, f2, ..., f8, alpha, beta]
        u_thrust = np.random.uniform(0, self.config['max_thrust'], 8)
        u_servo = np.random.uniform(-self.config['servo_range'], 
                                     self.config['servo_range'], 2)
        return np.concatenate([u_thrust, u_servo])
    
    def distance(self, x1, x2):
        """计算状态空间距离 (位置为主，速度为辅)"""
        pos_dist = np.linalg.norm(x1[:3] - x2[:3])
        vel_dist = np.linalg.norm(x1[7:10] - x2[7:10])
        return self.config['w_pos'] * pos_dist + self.config['w_vel'] * vel_dist
    
    def check_collision(self, state):
        """碰撞检测"""
        x, y, z = state[:3]
        
        # 边界检测
        if not (self.bounds[0] <= x <= self.bounds[1] and 
                self.bounds[2] <= y <= self.bounds[3] and 
                self.bounds[4] <= z <= self.bounds[5]):
            return True  # 出界
        
        # TODO: 障碍物检测
        for obs in self.obstacles:
            # obs: {'center': [x, y, z], 'radius': r}
            if np.linalg.norm(state[:3] - obs['center']) < obs['radius']:
                return True
        
        return False
    
    def plan(self, verbose=True):
        """执行 RRT 规划"""
        max_iter = self.config['max_iter']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始 Kinodynamic RRT 规划")
            print(f"起点: {self.start[:3]}")
            print(f"终点: {self.goal}")
            print(f"最大迭代次数: {max_iter}")
            print(f"{'='*60}\n")
        
        for i in range(max_iter):
            # 1. 采样 (偏向目标采样)
            if np.random.rand() < self.config['goal_sample_rate']:
                # 目标状态 (位置为目标，速度为0，姿态为单位四元数)
                x_rand = np.concatenate([
                    self.goal,                    # [px, py, pz]
                    [1, 0, 0, 0],                 # [qw, qx, qy, qz]
                    np.zeros(6)                   # [vx, vy, vz, wx, wy, wz]
                ])
            else:
                # 随机采样
                x_rand = np.array([
                    np.random.uniform(self.bounds[0], self.bounds[1]),  # px
                    np.random.uniform(self.bounds[2], self.bounds[3]),  # py
                    np.random.uniform(self.bounds[4], self.bounds[5]),  # pz
                    1, 0, 0, 0,  # qw, qx, qy, qz (简化：不随机采样姿态)
                    *np.random.uniform(-2, 2, 6)  # vx, vy, vz, wx, wy, wz
                ])
            
            # 2. 找最近邻
            dists = [self.distance(node['state'], x_rand) for node in self.tree]
            nearest_idx = np.argmin(dists)
            x_near = self.tree[nearest_idx]['state']
            
            # 3. Kinodynamic 扩展
            best_x_new = None
            best_control = None
            min_dist_to_rand = float('inf')
            
            for _ in range(self.config['num_control_samples']):
                u_try = self.get_random_control()
                
                try:
                    # 动力学积分
                    res = self.integrator(x0=x_near, u=u_try)
                    x_new = np.array(res['xf']).flatten()
                    
                    # 四元数归一化
                    q_norm = np.linalg.norm(x_new[3:7])
                    if q_norm > 1e-6:
                        x_new[3:7] /= q_norm
                    
                    # 边界和碰撞检测
                    if not self.check_collision(x_new):
                        d = self.distance(x_new, x_rand)
                        if d < min_dist_to_rand:
                            min_dist_to_rand = d
                            best_x_new = x_new
                            best_control = u_try
                except Exception as e:
                    # 积分失败，跳过
                    continue
            
            # 4. 添加新节点到树
            if best_x_new is not None:
                new_cost = self.tree[nearest_idx]['cost'] + self.distance(x_near, best_x_new)
                new_node = {
                    'state': best_x_new,
                    'parent': nearest_idx,
                    'cost': new_cost,
                    'control': best_control
                }
                self.tree.append(new_node)
                
                # 5. 检查是否到达目标
                dist_to_goal = np.linalg.norm(best_x_new[:3] - self.goal)
                if dist_to_goal < self.config['goal_tolerance']:
                    if verbose:
                        print(f"\n✓ 找到路径!")
                        print(f"  迭代次数: {i+1}/{max_iter}")
                        print(f"  树节点数: {len(self.tree)}")
                        print(f"  路径代价: {new_cost:.3f}")
                        print(f"  终点误差: {dist_to_goal:.3f}m\n")
                    
                    return self.extract_path(len(self.tree) - 1)
            
            # 进度显示
            if verbose and (i+1) % 100 == 0:
                print(f"迭代 {i+1}/{max_iter}, 树节点数: {len(self.tree)}")
        
        if verbose:
            print(f"\n✗ 未找到路径 (达到最大迭代次数 {max_iter})")
            print(f"  树节点数: {len(self.tree)}")
            print(f"  最近距离: {min([np.linalg.norm(n['state'][:3] - self.goal) for n in self.tree]):.3f}m\n")
        
        return None
    
    def extract_path(self, end_idx):
        """提取路径 (从起点到终点)"""
        path_states = []
        path_controls = []
        
        curr = end_idx
        while curr is not None:
            path_states.append(self.tree[curr]['state'])
            if self.tree[curr]['control'] is not None:
                path_controls.append(self.tree[curr]['control'])
            curr = self.tree[curr]['parent']
        
        # 倒序 (起点 -> 终点)
        path_states = path_states[::-1]
        path_controls = path_controls[::-1]
        
        return {
            'states': path_states,
            'controls': path_controls,
            'num_waypoints': len(path_states),
            'total_cost': self.tree[end_idx]['cost']
        }


def test_planner():
    """测试规划器"""
    # 初始状态: 原点，静止
    start_state = np.array([0, 0, 0.5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # 目标位置
    goal_pos = np.array([2, 2, 1])
    
    # 边界
    bounds = [-5, 5, -5, 5, 0.2, 3]
    
    # 创建规划器
    planner = KinodynamicRRT(start_state, goal_pos, bounds)
    
    # 执行规划
    result = planner.plan()
    
    if result:
        print(f"路径节点数: {result['num_waypoints']}")
        print(f"路径代价: {result['total_cost']:.3f}")
        print("\n路径节点 (前5个):")
        for i, state in enumerate(result['states'][:5]):
            print(f"  [{i}] pos: ({state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f})")


if __name__ == '__main__':
    test_planner()
