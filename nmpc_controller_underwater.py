# 四旋翼NMPC控制器 20250304 Wakkk
# ACADOS NMPC
from acados_template import AcadosOcp, AcadosOcpSolver
from export_model import *
from export_model_underwater import *
import numpy as np
import scipy.linalg
from os.path import dirname, join, abspath
import time

# np.set_printoptions(precision=3)  # 设置精度
np.set_printoptions(suppress=True)  # 禁用科学计数法输出

# ACADOS NMPC控制器
class NMPC_Controller:
    def __init__(self):
        self.ocp = AcadosOcp()       # OCP 优化问题
        # self.model = export_model()  # 导出四旋翼物理模型
        self.model = export_model_underwater()
        
        # 数据记录
        self.data_log = []

        self.Tf = 0.5                       # 预测时间长度(s) - 降低以提高求解速度
        self.N = 30                         # 预测步数(节点数量) - 降低以提高稳定性
        self.nx = self.model.x.size()[0]    # 状态维度 13维度
        self.nu = self.model.u.size()[0]    # 控制输入维度 4维度
        self.ny = self.nx + self.nu         # 评估维度
        self.ny_e = self.nx                 # 终端评估维度

        # set ocp_nlp_dimensions
        self.nlp_dims     = self.ocp.dims
        self.nlp_dims.N   = self.N

        # parameters
        self.g0  = 9.8066    # [m.s^2] accerelation of gravity
        self.mq  = 4.672     # [kg] total mass (with one marker)
        self.Ct  = 0.1757   # [N/krpm^2] Thrust coef

        # bounds
        self.hov_w = 2 * np.sqrt((self.mq*self.g0)/(4*self.Ct))  # 悬停时单个电机推力 1 3 5 7
        print(f"hovor speed: {self.hov_w} krpm")
        # hovor speed: 15.777730167256925 krpm

        self.max_speed = 30.0  # 电机最高转速(krpm) - 增加以提供更大控制余量

        # set weighting matrices 状态权重矩阵
        Q = np.eye(self.nx)
        Q[0,0] = 60.0        # x - 适度降低，优先保持姿态稳定
        Q[1,1] = 60.0        # y
        Q[2,2] = 150.0       # z - 增加以减小稳态误差
        # 姿态权重 (约束飞行器保持水平姿态) - 极其重要！
        Q[3,3] = 0.0         # qw (标量部分，通常不直接约束)
        Q[4,4] = 60.0        # qx (roll) - 增加以提高抗干扰性
        Q[5,5] = 80.0        # qy (pitch) - 增加以提高抗干扰性
        Q[6,6] = 50.0        # qz (yaw) - 增加以减小yaw误差
        Q[7,7] = 10.0        # vbx - 适度约束速度
        Q[8,8] = 10.0        # vby
        Q[9,9] = 30.0        # vbz - 降低以允许更大速度响应
        Q[10,10] = 30.0      # wx - 增加以提高抗干扰性
        Q[11,11] = 40.0      # wy - 增加以提高抗干扰性
        Q[12,12] = 100.0     # wz - 大幅增加以改善yaw控制

        R = np.eye(self.nu)   # 控制输入权重矩阵
        R[0,0] = 0.5         # w_front - 适度约束，避免控制过激
        R[1,1] = 0.5         # w_left
        R[2,2] = 0.5         # w_rear
        R[3,3] = 0.5         # w_right
        R[4,4] = 0.02        # yaw_bias - 降低以允许更激进的yaw控制

        self.ocp.cost.W = scipy.linalg.block_diag(Q, R)

        Vx = np.zeros((self.ny, self.nx))
        Vx[0,0] = 1.0
        Vx[1,1] = 1.0
        Vx[2,2] = 1.0
        Vx[3,3] = 1.0
        Vx[4,4] = 1.0
        Vx[5,5] = 1.0
        Vx[6,6] = 1.0
        Vx[7,7] = 1.0
        Vx[8,8] = 1.0
        Vx[9,9] = 1.0
        Vx[10,10] = 1.0
        Vx[11,11] = 1.0
        Vx[12,12] = 1.0
        self.ocp.cost.Vx = Vx

        Vu = np.zeros((self.ny, self.nu))
        Vu[13,0] = 1.0
        Vu[14,1] = 1.0
        Vu[15,2] = 1.0
        Vu[16,3] = 1.0
        Vu[17,4] = 1.0
        self.ocp.cost.Vu = Vu

        self.ocp.cost.W_e = 50.0 * Q

        Vx_e = np.zeros((self.ny_e, self.nx))
        Vx_e[0,0] = 1.0
        Vx_e[1,1] = 1.0
        Vx_e[2,2] = 1.0
        Vx_e[3,3] = 1.0
        Vx_e[4,4] = 1.0
        Vx_e[5,5] = 1.0
        Vx_e[6,6] = 1.0
        Vx_e[7,7] = 1.0
        Vx_e[8,8] = 1.0
        Vx_e[9,9] = 1.0
        Vx_e[10,10] = 1.0
        Vx_e[11,11] = 1.0
        Vx_e[12,12] = 1.0
        self.ocp.cost.Vx_e = Vx_e

        # 过程参考向量(状态+输入)
        self.ocp.cost.yref   = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.hov_w, self.hov_w, self.hov_w, self.hov_w, 0.0])
        # 终端参考向量(状态)
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 构建约束
        self.ocp.constraints.lbu = np.array([0.0, 0.0, 0.0, 0.0, -0.8])  # yaw_bias限制在±0.5（配合k_yaw=0.3，允许±15%调整）
        self.ocp.constraints.ubu = np.array([+self.max_speed,+self.max_speed,+self.max_speed,+self.max_speed, 0.8])  # 避免过度不平衡
        self.ocp.constraints.x0  = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 初始状态
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])  # 所有电机转速参与评估

        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'  
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.print_level = 0

        # set prediction horizon
        self.ocp.solver_options.tf = self.Tf
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # 显然更快 ~100Hz
        # self.ocp.solver_options.nlp_solver_type = 'SQP'  # ~10Hz

        self.ocp.model = self.model  # 传入模型
        
        # 设置参数初始值 (扰动力和力矩: [fx, fy, fz, mx, my, mz])
        self.ocp.parameter_values = np.zeros(6)  # 默认零扰动
        
        # 构建编译OCP求解器
        self.acados_solver = AcadosOcpSolver(self.ocp, json_file = 'acados_ocp.json')
        print("NMPC Controller Init Done")

    # 状态空间位点控制
    # current_state当前状态: [x, y, z, qw, qx, qy, qz, vbx, vby, vbz, wx, wy, wz] 
    # goal_state目标状态:    [x, y, z, qw, qx, qy, qz, vbx, vby, vbz, wx, wy, wz]
    # disturbance扰动估计: [fx, fy, fz, mx, my, mz] (可选)
    def nmpc_state_control(self, current_state, goal_state, disturbance=None):
        _start = time.perf_counter()
        # Set initial condition, equality constraint
        self.acados_solver.set(0, 'lbx', current_state)
        self.acados_solver.set(0, 'ubx', current_state)

        y_ref = np.concatenate((goal_state, np.array([self.hov_w, self.hov_w, self.hov_w, self.hov_w, 0.0])))
        # Set Goal State
        for i in range(self.N):
            self.acados_solver.set(i, 'yref', y_ref)   # 过程参考
        y_refN = goal_state 
        self.acados_solver.set(self.N, 'yref', y_refN)   # 终端参考

        # 设置扰动参数 (如果提供)
        if disturbance is not None:
            # disturbance = [fx, fy, fz, mx, my, mz]
            p = disturbance
        else:
            # 默认零扰动
            p = np.zeros(6)
        
        # 为所有预测步长设置扰动参数
        for i in range(self.N):
            self.acados_solver.set(i, 'p', p)

        # Solve Problem
        status = self.acados_solver.solve()
        
        # 检查求解器状态
        if status != 0:
            print(f"⚠️ ACADOS求解失败! status={status}")
            if status == 1:
                print("   → 求解器达到最大迭代次数")
            elif status == 2:
                print("   → 求解器初始化失败")
            elif status == 3:
                print("   → QP求解器失败")
            elif status == 4:
                print("   → 求解器达到最大迭代次数（未收敛）")
            # 返回悬停控制量作为安全回退
            _end = time.perf_counter()
            _dt = _end - _start
            return _dt, np.array([self.hov_w, self.hov_w, self.hov_w, self.hov_w, 0.0])
        
        # Get Solution (仅在成功时获取)
        w_opt_acados = np.ndarray((self.N, 5))  # 控制输入
        x_opt_acados = np.ndarray((self.N + 1, len(current_state)))   # 状态估计
        x_opt_acados[0, :] = self.acados_solver.get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_solver.get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_solver.get(i + 1, "x")
        # return w_opt_acados, x_opt_acados  # 返回控制输入和状态
        _end = time.perf_counter()
        _dt = _end - _start
        return _dt, w_opt_acados[0]  # 返回最近控制输入 4 Vector
        # control_input = self.acados_solver.get(0, "u")
        # state_estimate = self.acados_solver.get(self.N, "x")
        # return control_input, state_estimate  # 返回所有控制输入和状态

    # NMPC位置控制
    # goal_pos: 目标三维位置[x y z]
    def nmpc_position_control(self, current_state, goal_pos, disturbance=None):
        """
        位置控制
        
        Args:
            current_state: 当前状态 [x,y,z,qw,qx,qy,qz,vx,vy,vz,wx,wy,wz]
            goal_pos: 目标位置 [x,y,z]
            disturbance: 扰动估计 dict {'force': [fx,fy,fz], 'torque': [mx,my,mz]}
                        可选，直接用于MPC模型中
        
        Returns:
            _dt: 求解时间
            control: 控制输出 [w_front, w_left, w_rear, w_right, yaw_bias]
        """
        goal_state = np.array([goal_pos[0], goal_pos[1], goal_pos[2], 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 将扰动字典转换为参数数组 [fx, fy, fz, mx, my, mz]
        if disturbance is not None:
            dist_array = np.concatenate([disturbance['force'], disturbance['torque']])
        else:
            dist_array = None
        
        # 将扰动参数传递给MPC求解器
        _dt, control = self.nmpc_state_control(current_state, goal_state, dist_array)
        
        # 记录数据: [time, state(13), goal(3), control(5), solve_time, disturbance]
        import time
        log_entry = {
            'time': time.time(),
            'state': current_state.copy(),
            'goal': goal_pos.copy(),
            'control': control.copy(),
            'solve_time': _dt
        }
        
        # 记录扰动估计（如果有）
        if disturbance is not None:
            log_entry['dist_force'] = disturbance['force'].copy()
            log_entry['dist_torque'] = disturbance['torque'].copy()
        else:
            log_entry['dist_force'] = np.zeros(3)
            log_entry['dist_torque'] = np.zeros(3)
        
        self.data_log.append(log_entry)
        
        return _dt, control
    
    def save_data(self, filename='nmpc_data.csv'):
        """保存数据到CSV文件"""
        import csv
        if not self.data_log:
            print("没有数据可保存")
            return
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头（包含扰动估计）
            header = ['time', 
                     'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 
                     'vx', 'vy', 'vz', 'wx', 'wy', 'wz',
                     'goal_x', 'goal_y', 'goal_z',
                     'u_front', 'u_left', 'u_rear', 'u_right', 'u_yaw_bias',
                     'solve_time',
                     'dist_fx', 'dist_fy', 'dist_fz',
                     'dist_mx', 'dist_my', 'dist_mz']
            writer.writerow(header)
            
            # 写入数据
            t0 = self.data_log[0]['time']
            for data in self.data_log:
                row = [data['time'] - t0]  # 相对时间
                row.extend(data['state'])
                row.extend(data['goal'])
                row.extend(data['control'])
                row.append(data['solve_time'])
                # 添加扰动估计
                row.extend(data['dist_force'])
                row.extend(data['dist_torque'])
                writer.writerow(row)
        
        print(f"✓ 数据已保存到: {filename} ({len(self.data_log)} 条记录)")
    
    def clear_data(self):
        """清空数据日志"""
        self.data_log = []

# TEST
if __name__ == '__main__':
    print("20250304 ACADOS NMPC TEST")
    nmpc_controller = NMPC_Controller()
    current_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_state = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _dt, w = nmpc_controller.nmpc_state_control(current_state, goal_state)

    # print("Control Input:")
    # print(w)
    # print("State Estimation:")
    # print(x)
