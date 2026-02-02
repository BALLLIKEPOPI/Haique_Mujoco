#!/usr/bin/env python3
"""
Thrust Sensor Calibration Script
Read log/推力数据.xlsx, perform linear fitting, and output calibration formula and plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

def load_thrust_data(file_path):
    """Load thrust sensor data"""
    df = pd.read_excel(file_path)
    
    # Extract actual weights (unit: g)
    actual_weights = df['实际重量g'].values
    
    # Extract all 6 measurements from sensor
    sensor_cols = [col for col in df.columns if '传感器数据第' in col]
    sensor_data = df[sensor_cols].values  # shape: (n_samples, 6)
    
    return actual_weights, sensor_data

def linear_calibration(actual_weights, sensor_data):
    """
    Linear calibration: actual_weight(g) = k * sensor_value + b
    
    Args:
        actual_weights: Array of actual weights (n_samples,)
        sensor_data: Array of sensor data (n_samples, n_measurements)
    
    Returns:
        k, b: Linear fitting parameters
        r_squared: Coefficient of determination R²
        sensor_mean: Mean sensor value for each weight
        sensor_std: Standard deviation for each weight
    """
    # Calculate mean and std for each weight
    sensor_mean = np.mean(sensor_data, axis=1)
    sensor_std = np.std(sensor_data, axis=1)
    
    # Linear regression: actual_weight = k * sensor_value + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(sensor_mean, actual_weights)
    
    r_squared = r_value ** 2
    
    return slope, intercept, r_squared, sensor_mean, sensor_std

def plot_calibration(actual_weights, sensor_mean, sensor_std, k, b, r_squared, save_path):
    """
    Plot calibration curve
    
    Args:
        actual_weights: Array of actual weights
        sensor_mean: Mean sensor values
        sensor_std: Standard deviation of sensor values
        k, b: Fitting parameters
        r_squared: R² value
        save_path: Save path for the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== Left plot: Linear fitting curve =====
    ax1.errorbar(sensor_mean, actual_weights, xerr=sensor_std, 
                 fmt='o', color='steelblue', markersize=8, 
                 capsize=5, label='Experimental Data (±1σ)', alpha=0.7)
    
    # Plot fitting line
    sensor_range = np.linspace(sensor_mean.min(), sensor_mean.max(), 100)
    weight_fit = k * sensor_range + b
    ax1.plot(sensor_range, weight_fit, 'r--', linewidth=2, 
             label=f'Linear Fit: y = {k:.4f}x + {b:.2f}')
    
    ax1.set_xlabel('Sensor Output Value', fontsize=12)
    ax1.set_ylabel('Actual Weight (g)', fontsize=12)
    ax1.set_title(f'Thrust Sensor Calibration Curve (R² = {r_squared:.6f})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== Right plot: Residual analysis =====
    predicted_weights = k * sensor_mean + b
    residuals = actual_weights - predicted_weights
    
    ax2.scatter(predicted_weights, residuals, color='coral', s=80, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Predicted Weight (g)', fontsize=12)
    ax2.set_ylabel('Residuals (g)', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add residual statistics
    rmse = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))
    textstr = f'RMSE = {rmse:.2f} g\nMax Error = {max_error:.2f} g'
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Calibration plot saved: {save_path}")
    plt.close()  # Close to avoid display warnings

def save_calibration_formula(k, b, r_squared, rmse, save_path):
    """Save calibration formula to text file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Thrust Sensor Calibration Formula\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Linear Fitting Model:\n")
        f.write(f"  actual_weight(g) = {k:.6f} × sensor_value + {b:.4f}\n\n")
        
        f.write("Fitting Quality:\n")
        f.write(f"  R² (coefficient of determination) = {r_squared:.8f}\n")
        f.write(f"  RMSE (root mean square error) = {rmse:.4f} g\n\n")
        
        f.write("Usage:\n")
        f.write("  sensor_value = <sensor reading>\n")
        f.write(f"  actual_weight_g = {k:.6f} * sensor_value + {b:.4f}\n\n")
        
        f.write("Convert to Newtons (N):\n")
        f.write("  force_N = actual_weight_g * 9.8 / 1000\n")
        f.write(f"  force_N = ({k:.6f} * sensor_value + {b:.4f}) * 0.0098\n\n")
        
        # Simplified formula
        k_newton = k * 0.0098
        b_newton = b * 0.0098
        f.write("Simplified Formula (direct output in Newtons):\n")
        f.write(f"  force_N = {k_newton:.8f} * sensor_value + {b_newton:.6f}\n\n")
        
        f.write("=" * 60 + "\n")
    
    print(f"✓ Calibration formula saved: {save_path}")

def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "log" / "推力数据.xlsx"
    output_dir = base_dir / "log" / "pic"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if not exists
    
    print("=" * 60)
    print("Thrust Sensor Calibration Program")
    print("=" * 60)
    
    # 1. Load data
    print(f"\n📂 Loading data: {data_file}")
    actual_weights, sensor_data = load_thrust_data(data_file)
    print(f"   Number of data points: {len(actual_weights)}")
    print(f"   Weight range: {actual_weights.min():.1f} ~ {actual_weights.max():.1f} g")
    
    # 2. Linear fitting
    print("\n🔬 Performing linear fitting...")
    k, b, r_squared, sensor_mean, sensor_std = linear_calibration(actual_weights, sensor_data)
    
    # Calculate RMSE
    predicted_weights = k * sensor_mean + b
    residuals = actual_weights - predicted_weights
    rmse = np.sqrt(np.mean(residuals**2))
    max_error = np.max(np.abs(residuals))
    
    # 3. Output results
    print("\n" + "=" * 60)
    print("📊 Calibration Results")
    print("=" * 60)
    print(f"\nLinear Fitting Formula:")
    print(f"  actual_weight(g) = {k:.6f} × sensor_value + {b:.4f}")
    print(f"\nFitting Quality:")
    print(f"  R² = {r_squared:.8f}  {'✓ Excellent' if r_squared > 0.99 else '⚠️ Needs improvement'}")
    print(f"  RMSE = {rmse:.4f} g")
    print(f"  Max Error = {max_error:.4f} g")
    
    # Convert to Newtons
    k_newton = k * 0.0098
    b_newton = b * 0.0098
    print(f"\nDirect conversion to force (N):")
    print(f"  force_N = {k_newton:.8f} × sensor_value + {b_newton:.6f}")
    
    # 4. Save results
    print("\n💾 Saving results...")
    plot_path = output_dir / "thrust_calibration_curve.png"
    formula_path = output_dir / "thrust_calibration_formula.txt"
    
    plot_calibration(actual_weights, sensor_mean, sensor_std, k, b, r_squared, plot_path)
    save_calibration_formula(k, b, r_squared, rmse, formula_path)
    
    # 5. Display examples
    print("\n" + "=" * 60)
    print("📝 Usage Examples")
    print("=" * 60)
    example_sensor = [-3000, -5000, -7000]
    for sv in example_sensor:
        weight_g = k * sv + b
        force_n = weight_g * 0.0098
        print(f"  Sensor value = {sv:6d}  =>  {weight_g:7.2f} g  =  {force_n:6.3f} N")
    
    print("\n" + "=" * 60)
    print("✅ Calibration Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
