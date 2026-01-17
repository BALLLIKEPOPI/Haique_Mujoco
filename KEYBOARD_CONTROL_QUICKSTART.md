# ⌨️ 键盘控制系统 - 快速开始

## 📂 文件结构

```
keyboard_controller.py        # 键盘输入节点
controller_state_machine.py   # 状态机（悬停/运动模式切换）
main_keyboard.py              # 主程序
```

## 🚀 快速启动（3步）

### 1. 测试键盘输入
```bash
python3 keyboard_controller.py
```
弹出窗口后按**W/A/S/D**测试，看到速度输出说明正常。

### 2. 运行完整控制
```bash
python3 main_keyboard.py
```

### 3. 开始操作
- 按**H**查看帮助
- 按**W/S**前进/后退
- 按**A/D**左移/右移
- 按**空格**上升，**Shift**下降
- 按**ESC**紧急停止

## ⌨️ 键盘映射表

| 按键 | 功能 | 最大速度 |
|------|------|---------|
| W | 前进 | 1.5 m/s |
| S | 后退 | -1.5 m/s |
| A | 左移 | -1.5 m/s |
| D | 右移 | 1.5 m/s |
| 空格 | 上升 | 0.8 m/s |
| Shift | 下降 | -0.8 m/s |
| Q | 偏航左 | -1.0 rad/s |
| E | 偏航右 | 1.0 rad/s |
| **ESC** | **紧急停止** | - |
| **R** | **解除停止** | - |
| H | 帮助信息 | - |

## 🎯 控制模式

### 悬停模式（自动）
- 触发条件：水平速度 < 0.05 m/s
- 使用控制器：`NMPCController`（4电机）
- 特点：高精度位置保持，舵机固定0°

### 运动模式（自动）
- 触发条件：水平速度 ≥ 0.05 m/s  
- 使用控制器：`NMPCControllerUnderwater`（8电机）
- 特点：高机动性，舵机可调

**状态机自动切换，0.3秒延迟防抖**

## 📊 参数配置

在 `main_keyboard.py` 中修改：
```python
keyboard = KeyboardController(
    max_vel_xy=1.5,      # 最大水平速度 (m/s)
    max_vel_z=0.8,       # 最大垂直速度 (m/s)
    max_yaw_rate=1.0,    # 最大偏航速度 (rad/s)
    accel_rate=2.0       # 加速度 (m/s²)
)
```

## ❓ 常见问题

**Q: pygame窗口无法显示？**
```bash
export DISPLAY=:0
python3 keyboard_controller.py
```

**Q: 按键无响应？**
- 确保pygame窗口处于激活状态（点击窗口）

**Q: 想调整速度范围？**
- 修改 `main_keyboard.py` 中 `KeyboardController()` 参数

## ✅ 测试清单

- [ ] pygame窗口能正常弹出
- [ ] 按W/A/S/D能看到速度输出
- [ ] 松开按键速度归零
- [ ] 按ESC能紧急停止
- [ ] 按R能解除停止

## 📈 性能特性

- ✅ 50Hz更新频率
- ✅ 平滑加速（2.0 m/s²）
- ✅ 实时速度显示
- ✅ 紧急停止保护
- ✅ 自动模式切换

---

**详细文档**: [KEYBOARD_CONTROL_README.md](KEYBOARD_CONTROL_README.md)
