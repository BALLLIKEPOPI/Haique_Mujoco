#!/usr/bin/env python3
"""
Thrust Data Quadratic Curve Fitting
Using nonlinear least squares (scipy.curve_fit) to fit thrust vs RPM
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# Configure plot style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Path configuration
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == "visualization" else script_dir
output_dir = project_root / "log" / "pic"
output_dir.mkdir(parents=True, exist_ok=True)

# Lever arm for anti-torque test (220 mm)
ARM_LENGTH_M = 0.22

# Data files to process
data_files = [
    ("log/单电机空中推力测试.xlsx", "Single Motor - Thrust", "推力（g）", "thrust"),
    ("log/单电机空中反扭测试.xlsx", "Single Motor - Torque", "反扭（g）", "torque"),
    ("log/双电机空中推力测试.xlsx", "Dual Motor - Thrust", "推力（g）", "thrust")
]

# Quadratic model: F = a*RPM^2 + b*RPM + c
def quadratic_model(rpm, a, b, c):
    """Quadratic thrust/torque model"""
    return a * rpm**2 + b * rpm + c

# Storage for all fitting results
all_results = []

# Process each data file
for data_file, title, force_column, quantity_type in data_files:
    data_path = project_root / data_file
    
    if not data_path.exists():
        print(f"⚠️  File not found: {data_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {title}")
    print(f"{'='*60}")
    
    # Load data
    print(f"Loading data: {data_path}")
    df = pd.read_excel(data_path, sheet_name=0)
    
    # Extract data
    raw_g = df[force_column].values  # Measured gram-force equivalent
    rpm = df['转速（rpm）'].values     # Motor speed in RPM

    # Convert measurements to physical quantity
    # thrust: g -> N
    # torque: g -> N, then T = F * L (L = 0.22 m)
    if quantity_type == "torque":
        output_val = raw_g * 0.00981 * ARM_LENGTH_M
        y_label = "Torque"
        y_unit = "N·m"
        symbol = "T"
    else:
        output_val = raw_g * 0.00981
        y_label = "Thrust"
        y_unit = "N"
        symbol = "F"
    
    print(f"\nData Overview:")
    print(f"  Samples: {len(raw_g)}")
    print(f"  Raw range: {raw_g.min():.1f} - {raw_g.max():.1f} g")
    print(f"  RPM range: {rpm.min():.0f} - {rpm.max():.0f} rpm")
    if quantity_type == "torque":
        print(f"  Lever arm: {ARM_LENGTH_M:.3f} m")
    
    # Nonlinear least squares fitting (scipy.curve_fit)
    print("\n=== Nonlinear Least Squares Fitting ===")
    params, covariance = curve_fit(quadratic_model, rpm, output_val, p0=[1e-6, 1e-3, 0])
    a, b, c = params
    print(f"Fitted Coefficients:")
    print(f"  a = {a:.6e} ({y_unit}/rpm²)")
    print(f"  b = {b:.6e} ({y_unit}/rpm)")
    print(f"  c = {c:.6e} ({y_unit})")
    
    # Calculate goodness of fit (R²)
    output_fit = quadratic_model(rpm, a, b, c)
    ss_res = np.sum((output_val - output_fit)**2)
    ss_tot = np.sum((output_val - np.mean(output_val))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nGoodness of Fit:")
    print(f"  R² = {r_squared:.6f}")
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((output_val - output_fit)**2))
    print(f"\nRMSE:")
    print(f"  RMSE = {rmse*1000:.2f} m{y_unit}")
    
    # Generate smooth curve for plotting
    rpm_smooth = np.linspace(rpm.min(), rpm.max(), 200)
    output_fit_smooth = quadratic_model(rpm_smooth, a, b, c)
    
    # Store results
    all_results.append({
        'title': title,
        'y_label': y_label,
        'y_unit': y_unit,
        'symbol': symbol,
        'rpm': rpm,
        'output_val': output_val,
        'rpm_smooth': rpm_smooth,
        'output_fit_smooth': output_fit_smooth,
        'output_fit': output_fit,
        'params': (a, b, c),
        'r_squared': r_squared,
        'rmse': rmse
    })

# ============================================================
# Create combined visualization (2 rows × 3 columns)
# ============================================================
if len(all_results) > 0:
    fig = plt.figure(figsize=(18, 10))
    
    # Plot each dataset
    for idx, result in enumerate(all_results):
        # Subplot for fitted curve (top row)
        ax1 = plt.subplot(2, 3, idx + 1)
        ax1.scatter(result['rpm'], result['output_val'], s=80, alpha=0.7, color='blue', 
                   edgecolors='black', linewidth=1.5, label='Measured Data', zorder=3)
        ax1.plot(result['rpm_smooth'], result['output_fit_smooth'], 'r-', linewidth=2.5, 
                label=f'Quadratic Fit (R²={result["r_squared"]:.4f})', alpha=0.8)
        ax1.set_xlabel('Motor Speed (rpm)', fontsize=11, fontweight='bold')
        ax1.set_ylabel(f'{result["y_label"]} ({result["y_unit"]})', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_title(f'{result["title"]}', fontsize=12, fontweight='bold')
        
        # Add formula annotation
        a, b, c = result['params']
        formula_str = f'{result["symbol"]} = {a:.2e}·RPM²\n    + {b:.2e}·RPM\n    + {c:.2e}'
        ax1.text(0.98, 0.05, formula_str, transform=ax1.transAxes, 
                fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Subplot for residual analysis (bottom row)
        ax2 = plt.subplot(2, 3, idx + 4)
        residuals = result['output_val'] - result['output_fit']
        ax2.scatter(result['rpm'], residuals*1000, s=80, alpha=0.7, color='red', 
                   edgecolors='black', linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Motor Speed (rpm)', fontsize=11, fontweight='bold')
        ax2.set_ylabel(f'Residual (m{result["y_unit"]})', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_title(f'Residual (RMSE={result["rmse"]*1000:.2f} m{result["y_unit"]})', fontsize=12, fontweight='bold')
        
        # Add statistics annotation
        stats_str = f'Mean: {residuals.mean()*1000:.2f} m{result["y_unit"]}\nStd: {residuals.std()*1000:.2f} m{result["y_unit"]}'
        ax2.text(0.02, 0.98, stats_str, transform=ax2.transAxes, 
                fontsize=8, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Motor Performance Quadratic Fitting Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save combined figure
    output_path = output_dir / 'motor_performance_combined_fit.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Combined figure saved: {output_path}")
    
    # Save all formulas to one text file
    formula_file = output_dir / 'motor_performance_formulas.txt'
    with open(formula_file, 'w', encoding='utf-8') as f:
        f.write("Motor Performance Quadratic Fitting Formulas\n")
        f.write("="*60 + "\n\n")
        
        for result in all_results:
            a, b, c = result['params']
            f.write(f"{result['title']}\n")
            f.write("-"*60 + "\n")
            f.write("F(RPM) = a·RPM² + b·RPM + c\n\n")
            f.write(f"Coefficients:\n")
            f.write(f"  a = {a:.10e} {result['y_unit']}/rpm²\n")
            f.write(f"  b = {b:.10e} {result['y_unit']}/rpm\n")
            f.write(f"  c = {c:.10e} {result['y_unit']}\n\n")
            f.write(f"Goodness of Fit: R² = {result['r_squared']:.8f}\n")
            f.write(f"RMSE = {result['rmse']*1000:.4f} m{result['y_unit']}\n\n")
            f.write(f"Python Code:\n")
            f.write(f"def {result['y_label'].lower()}_model(rpm):\n")
            f.write(f"    return {a:.10e} * rpm**2 + {b:.10e} * rpm + {c:.10e}\n\n")
            f.write("="*60 + "\n\n")
    
    print(f"✓ Formulas saved: {formula_file}")
    
    print(f"\n{'='*60}")
    print("All fitting completed!")
    print(f"{'='*60}")
    
    plt.show()
else:
    print("\n⚠️  No data files were successfully processed.")
