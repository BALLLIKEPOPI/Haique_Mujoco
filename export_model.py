# 导出四旋翼物理模型 20250304 Wakkk
from acados_template import AcadosModel
from casadi import SX, vertcat

def export_model(
    *,
    g0: float = 9.8066,
    mass: float = 4.672,
    inertia=(0.10170715, 0.10222875, 0.16095642),
    Ct: float = 0.1757,
    Cd: float = 0.02,
    dq: float = 0.605,
    k_yaw: float = 0.8,
):
    model_name = 'crazyflie'
    # parameters
    Ixx, Iyy, Izz = inertia
    l   = dq/2       # [m] distance between motors' center and the axis of rotation

    # 世界坐标系位置
    px = SX.sym('px')
    py = SX.sym('py')
    pz = SX.sym('pz')
    # 四元数
    q0 = SX.sym('q0')
    q1 = SX.sym('q1')
    q2 = SX.sym('q2')
    q3 = SX.sym('q3')
    # 世界坐标系速度
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')
    # 机体坐标系角速度
    wx = SX.sym('wx')
    wy = SX.sym('wy')
    wz = SX.sym('wz')
    # 构建状态向量
    x = vertcat(px, py, pz, q0, q1, q2, q3, vx, vy, vz, wx, wy, wz)

    # 系统控制输入: 四个电机的转速
    w1 = SX.sym('w1')
    w2 = SX.sym('w2')
    w3 = SX.sym('w3')
    w4 = SX.sym('w4')
    yaw_bias = SX.sym('yaw_bias')
    u = vertcat(w1, w2, w3, w4, yaw_bias)

    # for f_impl
    px_dot = SX.sym('px_dot')
    py_dot = SX.sym('py_dot')
    pz_dot = SX.sym('pz_dot')
    q0_dot = SX.sym('q0_dot')
    q1_dot = SX.sym('q1_dot')
    q2_dot = SX.sym('q2_dot')
    q3_dot = SX.sym('q3_dot')
    vx_dot = SX.sym('vx_dot')
    vy_dot = SX.sym('vy_dot')
    vz_dot = SX.sym('vz_dot')
    wx_dot = SX.sym('wx_dot')
    wy_dot = SX.sym('wy_dot')
    wz_dot = SX.sym('wz_dot')
    # 构建导数状态向量
    xdot = vertcat(px_dot, py_dot, pz_dot, q0_dot, q1_dot, q2_dot, q3_dot, vx_dot, vy_dot, vz_dot, wx_dot, wy_dot, wz_dot)

    # 扰动参数定义 (必须在使用前定义)
    dist_fx = SX.sym('dist_fx')  # 扰动力 x方向 (世界坐标系)
    dist_fy = SX.sym('dist_fy')  # 扰动力 y方向 (世界坐标系)
    dist_fz = SX.sym('dist_fz')  # 扰动力 z方向 (世界坐标系)
    dist_mx = SX.sym('dist_mx')  # 扰动力矩 x方向 (机体坐标系)
    dist_my = SX.sym('dist_my')  # 扰动力矩 y方向 (机体坐标系)
    dist_mz = SX.sym('dist_mz')  # 扰动力矩 z方向 (机体坐标系)

    # 位置求导
    px_d = vx
    py_d = vy
    pz_d = vz

    # Yaw coupling factor
    w1_ = w1 * (1 - k_yaw * yaw_bias)  # Front上,CW
    w2_ = w2 * (1 + k_yaw * yaw_bias)   # Left上,CCW
    w3_ = w3 * (1 - k_yaw * yaw_bias)   # Rear上,CW
    w4_ = w4 * (1 + k_yaw * yaw_bias)  # Right上,CCW
    w5_ = w1 * (1 + k_yaw * yaw_bias)  # Front下,CCW
    w6_ = w2 * (1 - k_yaw * yaw_bias)   # Left下,CW
    w7_ = w3 * (1 + k_yaw * yaw_bias)   # Rear下,CCW
    w8_ = w4 * (1 - k_yaw * yaw_bias)  # Right下,CW
    # 速度求导
    _thrust_acc_b = Ct*(w1_**2 + w2_**2 + w3_**2 + w4_**2 + w5_**2 + w6_**2 + w7_**2 + w8_**2) / mass  # 机体坐标系中推力引起的加速度
    # 将机体坐标系推力加速度转换为世界坐标系推力加速度
    # Rwb * [0, 0, _thrust_acc_b]
    _thrust_accx_w = 2*(q1*q3+q0*q2)*_thrust_acc_b
    _thrust_accy_w = 2*(-q0*q1+q2*q3)*_thrust_acc_b
    _thrust_accz_w = 2*(0.5-q1**2-q2**2)*_thrust_acc_b
    # 加入扰动力 (世界坐标系)
    vx_d = _thrust_accx_w + dist_fx / mass
    vy_d = _thrust_accy_w + dist_fy / mass
    vz_d = _thrust_accz_w - g0 + dist_fz / mass  # 重力加速度 + 扰动力

    # 四元数求导
    q0_d = -(q1*wx)/2 - (q2*wy)/2 - (q3*wz)/2
    q1_d =  (q0*wx)/2 - (q3*wy)/2 + (q2*wz)/2
    q2_d =  (q3*wx)/2 + (q0*wy)/2 - (q1*wz)/2
    q3_d =  (q1*wy)/2 - (q2*wx)/2 + (q0*wz)/2
    
    # 机体角速度求导
    # 计算三轴扭矩输入 (控制力矩)
    mx = Ct*l*(  w2_**2 + w6_**2 - w4_**2 - w8_**2)
    my = Ct*l*( -w1_**2 - w5_**2 + w3_**2 + w7_**2)
    mz = Cd*  ( -w1_**2 + w2_**2 - w3_**2 + w4_**2 + w5_**2 - w6_**2 + w7_**2 - w8_**2)
    # 计算角速度导数 (加入扰动力矩)
    wx_d = (mx + dist_mx + Iyy*wy*wz - Izz*wy*wz)/Ixx
    wy_d = (my + dist_my - Ixx*wx*wz + Izz*wx*wz)/Iyy
    wz_d = (mz + dist_mz + Ixx*wx*wy - Iyy*wx*wy)/Izz

    # Explicit and Implicit functions
    # 构建显式表达式和隐式表达式
    f_expl = vertcat(px_d, py_d, pz_d, q0_d, q1_d, q2_d, q3_d, vx_d, vy_d, vz_d, wx_d, wy_d, wz_d)
    f_impl = xdot - f_expl

    # algebraic variables
    z = []
    # parameters - 扰动力和力矩 (已在前面定义)
    p = vertcat(dist_fx, dist_fy, dist_fz, dist_mx, dist_my, dist_mz)
    # dynamics
    model = AcadosModel()  # 新建ACADOS模型

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name

    return model
