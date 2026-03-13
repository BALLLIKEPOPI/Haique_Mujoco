# Quadrotor-NMPC-Control 项目文件说明与使用指南

本文档按“目录 -> 文件”介绍项目中各部分的作用，以及常见使用方法。

## 1. 快速使用

### 1.1 运行水下主程序

```bash
python main_underwater.py
```

常见参数（以你的项目当前习惯为例）：

```bash
python main_underwater.py 1 --controller pid --no-eso
python main_underwater.py 1 --controller nmpc
```

### 1.2 运行空中主程序

```bash
python main.py
```

### 1.3 绘图脚本

```bash
python visualization/plot_nmpc_data.py
python visualization/plot_nmpc_underwater_data.py
python visualization/plot_pid_underwater_data.py
python visualization/fit_thrust_quadratic.py
```

输出图片默认保存在 `log/pic/`。

---

## 2. 项目顶层目录说明

### 2.1 第三方与生成代码

- `acados/`
  - 作用：acados 求解器源码/构建产物（第三方依赖）。
  - 用法：通常不直接改动，作为 NMPC 求解基础库。

- `c_generated_code/`
  - 作用：由 acados 导出的 C 求解器代码（含 `Makefile`、`*.c`、`*.h`、`*.so`）。
  - 用法：当模型或 OCP 改动后重新生成；Python 端控制器会调用对应求解器。

### 2.2 机器人与仿真资源

- `crazyfile/`
  - 作用：Crazyflie 相关 MJCF/XML 场景和资源。
  - 关键文件：`cf2.xml`、`scene.xml`、`assets/`。

- `robots/`
  - 作用：Haique 机体模型及网格。
  - 关键文件：`haique.xml`、`meshes/`。

- `images/`
  - 作用：README/文档中引用的示意图。

### 2.3 控制、模型、观测器

- `control/`
  - 作用：控制器实现（NMPC / PID / 键盘控制）。

- `model/`
  - 作用：动力学建模、模型导出、配置加载。

- `observer/`
  - 作用：ESO 等观测器与扰动生成模块。

- `observer_sim/`
  - 作用：观测器离线仿真、可视化与结果文件。

### 2.4 数据与可视化

- `log/`
  - 作用：日志、CSV、测试 Excel、输出图片。
  - 子目录：`csv/`、`npz/`、`pic/`。

- `visualization/`
  - 作用：绘图、标定与拟合脚本。

### 2.5 其他

- `__pycache__/`
  - 作用：Python 字节码缓存，可忽略。

- `.git/`
  - 作用：Git 元数据。

---

## 3. 顶层文件说明

- `README.md`
  - 作用：项目总体背景、模型与公式说明。
  - 用法：快速了解项目理论基础。

- `config.yaml`
  - 作用：控制/仿真参数配置。
  - 用法：调参数优先改这里，再运行主程序验证。

- `acados_ocp.json`
  - 作用：OCP 配置快照（权重、约束、时域等）。
  - 用法：与 NMPC 模型导出流程配合使用。

- `main.py`
  - 作用：空中场景主入口。
  - 用法：运行空中控制与仿真。

- `main_underwater.py`
  - 作用：水下场景主入口，支持状态机与控制器切换。
  - 用法：PID/NMPC/ESO 组合测试的核心脚本。

- `main_keyboard.py`
  - 作用：键盘交互控制入口。
  - 用法：人工遥控与记录控制日志。

- `controller_state_machine.py`
  - 作用：控制状态切换逻辑（如 HOVER/MOTION）。
  - 用法：由 `main_underwater.py` 调用。

- `trajectory_generator.py`
  - 作用：轨迹生成（目标位置/速度等）。
  - 用法：给控制器提供参考轨迹。

- `motor_failure_sim.py`
  - 作用：电机故障场景仿真入口。

- `visualize_motor_failure.py`
  - 作用：电机故障仿真结果可视化。

- `MUJOCO_LOG.TXT`
  - 作用：MuJoCo 运行日志。

- `control_performance_analysis.png`、`realistic_identification.png`
  - 作用：历史分析图片输出。

---

## 4. control/ 文件说明

- `control/nmpc_controller.py`
  - 作用：空中 NMPC 控制器。
  - 用法：由 `main.py` 或相关流程调用。

- `control/nmpc_controller_underwater.py`
  - 作用：水下 NMPC 控制器（8 电机 + 舵机等结构）。
  - 用法：`main_underwater.py --controller nmpc`。

- `control/pid_controller_underwater.py`
  - 作用：水下 PID 控制器（含分配逻辑、滤波、舵机控制）。
  - 用法：`main_underwater.py --controller pid`。

- `control/keyboard_controller.py`
  - 作用：键盘输入映射到控制指令。

---

## 5. model/ 文件说明

- `model/export_model.py`
  - 作用：导出空中模型（供 NMPC/acados 使用）。

- `model/export_model_underwater.py`
  - 作用：导出水下模型。

- `model/config_loader.py`
  - 作用：读取/解析 `config.yaml`。

---

## 6. observer/ 与 observer_sim/ 文件说明

### 6.1 observer/

- `observer/eso_observer.py`
  - 作用：ESO 观测器实现。

- `observer/disturbance_generator.py`
  - 作用：扰动注入/生成工具。

### 6.2 observer_sim/

- `observer_sim/simulation.py`
  - 作用：观测器仿真主脚本。

- `observer_sim/forward_dynamics.py`
  - 作用：前向动力学计算。

- `observer_sim/visualizer.py`、`observer_sim/visualize_fx_mz.py`
  - 作用：仿真结果可视化。

- `observer_sim/simulation_results.npz`
  - 作用：仿真中间数据。

- `observer_sim/*.png`
  - 作用：观测器仿真图。

---

## 7. visualization/ 文件说明

- `visualization/plot_nmpc_data.py`
  - 作用：空中 NMPC 日志绘图。

- `visualization/plot_nmpc_underwater_data.py`
  - 作用：水下 NMPC 日志绘图。

- `visualization/plot_pid_underwater_data.py`
  - 作用：水下 PID 日志绘图。

- `visualization/plot_keyboard_log.py`
  - 作用：键盘控制日志绘图。

- `visualization/plot_control_performance.py`
  - 作用：控制性能指标图。

- `visualization/calibrate_thrust_sensor.py`
  - 作用：推力传感器标定（线性映射）。

- `visualization/fit_thrust_quadratic.py`
  - 作用：推力/反扭二次拟合（最小二乘）并生成综合图。

- `visualization/lstsq_plot.py`
  - 作用：最小二乘相关绘图脚本（历史/实验用途）。

- `visualization/bayes_opts_plot.py`
  - 作用：贝叶斯优化结果可视化。

- `visualization/keyboard_control_plotter.py`
  - 作用：键控实验专用绘图。

---

## 8. log/ 数据文件说明

- `log/*.xlsx`
  - 作用：推力/反扭实验原始数据。
  - 常用文件：
    - `单电机空中推力测试.xlsx`
    - `单电机空中反扭测试.xlsx`
    - `双电机空中推力测试.xlsx`
    - `推力数据.xlsx`

- `log/*.csv`
  - 作用：控制运行日志（状态、目标、控制输出等）。

- `log/csv/`
  - 作用：结构化运行日志目录（绘图脚本默认读取位置之一）。

- `log/pic/`
  - 作用：所有可视化脚本输出目录。

- `log/npz/`
  - 作用：numpy 格式中间数据。

---

## 9. 常见工作流

### 9.1 跑控制并绘图

1. 运行控制：

```bash
python main_underwater.py 1 --controller pid --no-eso
```

2. 画图分析：

```bash
python visualization/plot_pid_underwater_data.py
```

### 9.2 更新模型并验证 NMPC

1. 导出模型（按场景选择脚本）。
2. 运行主程序生成日志。
3. 用 `plot_nmpc_data.py` 或 `plot_nmpc_underwater_data.py` 检查跟踪误差与控制输入。

### 9.3 推力/反扭建模

```bash
python visualization/fit_thrust_quadratic.py
```

查看输出：
- `log/pic/motor_performance_combined_fit.png`
- `log/pic/motor_performance_formulas.txt`

---

## 10. 维护建议

- 第三方目录 `acados/` 不建议直接改，优先在项目侧脚本中扩展。
- 生成目录 `c_generated_code/`、`__pycache__/` 以“可再生”为主，关注上游脚本改动。
- 参数修改优先集中在 `config.yaml` 与控制器实现文件，避免散落在多个入口脚本。

---

## 11. XML 文件说明（重点）

### 11.1 场景入口 XML

- `crazyfile/scene.xml`
  - 作用：仿真总场景入口（相机、光照、地面、统计等）。
  - 关键点：通过 `<include file="../robots/haique.xml"/>` 引入机体模型。
  - 使用方式：`main.py` 与 `main_underwater.py` 都通过该场景加载模型。

### 11.2 机体模型 XML

- `robots/haique.xml`
  - 作用：Haique 机体主体定义（刚体、惯量、关节、actuator、sensor、流体参数）。
  - 关键环境参数位置：文件开头 `<option ... density="..." viscosity="..." .../>`。
  - 当前示例中已提供两种配置：
    - 空气：`density="1.225" viscosity="1.8e-5"`
    - 水下（注释示例）：`density="1000" viscosity="0.001"`

### 11.3 Crazyflie 参考 XML

- `crazyfile/cf2.xml`
  - 作用：Crazyflie 参考模型。
  - 说明：同样有 `<option density="1.225" viscosity="1.8e-5"/>`，可作为空中参数参考。

---

## 12. 如何通过环境密度切换空中/水下仿真

本项目里“切换空中/水下”有两层：

1. **控制器与参数层**（Python 配置）
2. **物理介质层**（MuJoCo XML 的 density/viscosity）

两层建议同时切换，避免“空中参数跑水下介质”或“水下参数跑空气介质”的不一致。

### 12.1 控制器层切换（config + 入口脚本）

- 空中模式：`main.py` 使用 `get_mode_config("aerial")`
- 水下模式：`main_underwater.py` 使用 `get_mode_config("underwater")`
- 参数来源：`config.yaml` 中的 `aerial` / `underwater` 两套参数块

运行示例：

```bash
# 空中
python main.py

# 水下
python main_underwater.py 1 --controller nmpc
python main_underwater.py 1 --controller pid --no-eso
```

### 12.2 介质层切换（修改 robots/haique.xml）

在 `robots/haique.xml` 顶部找到 `<option .../>` 行，按需求切换：

```xml
<!-- Air -->
<option integrator="RK4" density="1.225" viscosity="1.8e-5" timestep="0.002"/>

<!-- Water -->
<!-- <option integrator="implicitfast" density="1000" viscosity="0.001" timestep="0.002"/> -->
```

建议改法：

1. 空中仿真：保留 Air 行，注释 Water 行。
2. 水下仿真：启用 Water 行，并注释 Air 行（或把 Air 参数改为水下值）。

### 12.3 推荐切换流程

1. 先改 `robots/haique.xml` 的 `density/viscosity`。
2. 再选对应入口脚本（`main.py` 或 `main_underwater.py`）。
3. 检查 `config.yaml` 是否使用对应模式参数。
4. 跑仿真并用绘图脚本验证响应是否符合预期。

### 12.4 常见问题

- 现象：控制发散或过慢。
  - 原因：介质参数和控制器参数不匹配。
  - 处理：确保 `density/viscosity` 与 `aerial/underwater` 控制参数成对使用。

- 现象：改了 `cf2.xml` 但结果没变。
  - 原因：主程序加载的是 `crazyfile/scene.xml`，其 include 的是 `robots/haique.xml`。
  - 处理：优先改 `robots/haique.xml`。
