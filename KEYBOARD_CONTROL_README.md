# ⌨️ 键盘控制模式 - WSL解决方案

## 问题背景

WSL2内核**不支持USB HID设备**（缺少`joydev`、`xpad`模块），无法使用游戏手柄。
因此提供键盘控制作为替代方案。

## 文件结构

```
keyboard_controller.py        # 键盘控制节点
main_keyboard.py              # 键盘控制主程序
controller_state_machine.py   # 状态机（与手柄版共用）
```

## 键盘映射

| 按键 | 功能 | 速度范围 |
|------|------|----------|
| **W** | 前进 (vx+) | 0 ~ 1.5 m/s |
| **S** | 后退 (vx-) | 0 ~ -1.5 m/s |
| **A** | 左移 (vy-) | 0 ~ -1.5 m/s |
| **D** | 右移 (vy+) | 0 ~ 1.5 m/s |
| **空格** | 上升 (vz+) | 0 ~ 0.8 m/s |
| **Shift** | 下降 (vz-) | 0 ~ -0.8 m/s |
| **Q** | 偏航左 | -1.0 rad/s |
| **E** | 偏航右 | +1.0 rad/s |
| **ESC** | 紧急停止 | 立即归零 |
| **R** | 解除停止 | 恢复控制 |
| **H** | 帮助信息 | - |

## 使用方法

### 1. 测试键盘输入（推荐先测试）

```bash
python3 keyboard_controller.py
```

会弹出pygame窗口，按W/A/S/D等键测试速度输出。

### 2. 运行完整控制系统

```bash
python3 main_keyboard.py
```

**注意**：需要先创建`controller_state_machine.py`！

### 3. 操作流程

1. 程序启动后会显示帮助信息
2. 按**H**随时查看键盘映射
3. 按**W/A/S/D**控制水平移动
4. 按**空格/Shift**控制垂直移动
5. 按**ESC**紧急停止，按**R**解除

## 状态机逻辑

- **悬停模式**（水平速度 < 0.05 m/s）
  - 使用`NMPCController`（4电机，舵机固定0）
  - 高精度位置保持
  
- **运动模式**（水平速度 ≥ 0.05 m/s）
  - 使用`NMPCControllerUnderwater`（8电机，舵机可调）
  - 高机动性水下运动

切换有0.3s延迟，避免抖动。

## 特性

- ✅ 平滑加速（2.0 m/s²加速度限制）
- ✅ 50Hz更新频率
- ✅ 紧急停止功能
- ✅ 实时速度指令显示
- ✅ 与状态机无缝集成

## 性能参数

```python
max_vel_xy = 1.5      # 最大水平速度 (m/s)
max_vel_z = 0.8       # 最大垂直速度 (m/s)
max_yaw_rate = 1.0    # 最大偏航速度 (rad/s)
accel_rate = 2.0      # 加速度 (m/s²)
```

## 故障排查

### pygame窗口无法显示

```bash
export DISPLAY=:0
python3 keyboard_controller.py
```

### 按键无响应

确保pygame窗口处于**焦点状态**（点击窗口激活）。

### 速度不平滑

检查CPU占用，确保50Hz循环正常运行。

## 对比：手柄 vs 键盘

| 特性 | 手柄 | 键盘 |
|------|------|------|
| 速度控制 | 模拟量（0-100%） | 离散（0/100%） |
| 精度 | 高 | 中等 |
| 易用性 | 直观 | 需要练习 |
| WSL支持 | ❌ 不支持 | ✅ 支持 |

## 未来升级

如果Microsoft更新WSL内核支持HID，可切换回手柄：

```bash
# 检查内核模块
modprobe joydev
# 如果成功，则可使用gamepad_controller.py
```

---

**推荐**：先用键盘测试功能，等迁移到原生Linux后再用手柄。
