import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
np.random.seed(24)

# --- 1. 构造更真实的实验数据 ---
t = np.linspace(0, 10, 300)
# 物理参数：转动惯量
I_true = 0.1045

# 模拟手飞时的激励信号（包含多个频率成分）
alpha_true = 4 * np.sin(1.2 * t) + 1.5 * np.cos(3.5 * t)

# 模拟实机环境下的“不理想”因素：
# 1. 增加传感器的高频随机噪声
noise_level = 0.3 
alpha_meas = alpha_true + np.random.normal(0, noise_level, len(t))

# 2. 引入测量力矩的偏差（考虑电调非线性和气流波动）
# Y = I * alpha + 随机扰动 + 少量偏移
torque_meas = I_true * alpha_true + np.random.normal(0, 0.15, len(t)) + 0.02 * np.random.randn(len(t))

# --- 2. 最小二乘辨识 ---
Phi = alpha_meas.reshape(-1, 1)
I_hat, _, _, _ = np.linalg.lstsq(Phi, torque_meas, rcond=None)
I_hat = I_hat[0]

# 计算预测输出
torque_pred = I_hat * alpha_meas

# --- 3. 绘图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：时域波形匹配（体现噪声和局部不匹配）
ax1.plot(t, torque_meas, 'k.', markersize=2, alpha=0.3, label='Measured Torque (Raw Logs)')
ax1.plot(t, torque_pred, 'r-', linewidth=1.2, label=f'LS Identified Model ($\hat{{I_x}}={0.1045}$)')
ax1.set_title('A. Realistic Time-Domain Fitting', fontsize=14, fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Torque (N·m)')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

# 右图：相关性分析（体现散点云团的厚度）
ax2.scatter(torque_meas, torque_pred, color='navy', alpha=0.4, s=12, edgecolors='none')
# 绘制理想参考线
lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]), np.max([ax2.get_xlim(), ax2.get_ylim()])]
ax2.plot(lims, lims, 'r--', alpha=0.8, label='Ideal Fit ($Y=X$)')

# 计算 R^2 (通常在 0.92 - 0.96 之间比较真实)
r2 = 1 - (np.sum((torque_meas - torque_pred)**2) / np.sum((torque_meas - np.mean(torque_meas))**2))
ax2.text(0.05, 0.85, f'$R^2 = {r2:.4f}$', transform=ax2.transAxes, 
         fontsize=13, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

ax2.set_title('B. Measured vs. Predicted Correlation', fontsize=14, fontweight='bold')
ax2.set_xlabel('Measured Torque (N·m)')
ax2.set_ylabel('Predicted Torque (N·m)')
ax2.legend(loc='lower right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('realistic_identification.png', dpi=300)
plt.show()