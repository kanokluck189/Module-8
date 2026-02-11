import numpy as np
import matplotlib.pyplot as plt
import random


def generate_data(n_points, true_c0=2.5, true_c1=1.8, noise_level=0.5, x_range=(-5, 5)):
    """
    Generate data points: y = true_c0 + true_c1*x + noise
    
    Args:
        n_points: Number of points to generate
        true_c0: True intercept
        true_c1: True slope
        noise_level: Standard deviation of Gaussian noise
        x_range: Range of x values
    """
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    noise = np.random.normal(0, noise_level, n_points)
    y = true_c0 + true_c1 * x + noise
    
    return x, y, true_c0, true_c1


def linear_regression_manual(x, y):
    """
    Perform linear regression by solving normal equations manually
    y = c0 + c1*x
    
    We solve: A^T * A * [c0, c1]^T = A^T * y
    where A = [1, x]
    """
    n = len(x)
    
    # Build normal equations manually
    # A^T * A = [[n, Σx], [Σx, Σx²]]
    # A^T * y = [Σy, Σxy]
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    # Build the 2x2 system
    ATA = np.array([[n, sum_x],
                    [sum_x, sum_x2]])
    
    ATy = np.array([sum_y, sum_xy])
    
    # Solve the system: (A^T * A) * c = A^T * y
    # Using manual 2x2 solution: c = (A^T * A)^(-1) * (A^T * y)
    det = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
    
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular, cannot solve")
    
    # Compute inverse manually
    ATA_inv = np.array([[ATA[1, 1], -ATA[0, 1]],
                        [-ATA[1, 0], ATA[0, 0]]]) / det
    
    # Solve for coefficients
    c = ATA_inv @ ATy
    c0, c1 = c[0], c[1]
    
    return c0, c1

def linear_regression_matrix(x, y):
    """
    Perform linear regression using matrix approach
    """
    n = len(x)
    
    # Build matrix A = [1, x]
    A = np.column_stack([np.ones(n), x])
    
    # Solve normal equations: A^T * A * c = A^T * y
    ATA = A.T @ A
    ATy = A.T @ y
    
    # Solve the system
    c = np.linalg.solve(ATA, ATy)
    
    return c[0], c[1]


def calculate_errors(x, y, c0, c1):
    """
    Calculate various error metrics
    """
    y_pred = c0 + c1 * x
    
    # Residuals
    residuals = y - y_pred
    
    # Mean Squared Error
    mse = np.mean(residuals**2)
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r_squared': r_squared,
        'residuals': residuals
    }


def plot_regression(x, y, c0, c1, true_c0, true_c1, title):
    """
    Plot data points, regression line, and true line
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, alpha=0.6, color='blue', label='Data points', s=30)
    
    # Plot regression line
    x_line = np.array([min(x), max(x)])
    y_reg = c0 + c1 * x_line
    plt.plot(x_line, y_reg, 'r-', linewidth=3, label=f'Regression: y = {c0:.3f} + {c1:.3f}x')
    
    # Plot true line (without noise)
    y_true = true_c0 + true_c1 * x_line
    plt.plot(x_line, y_true, 'g--', linewidth=2, label=f'True: y = {true_c0:.2f} + {true_c1:.2f}x')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add error metrics
    errors = calculate_errors(x, y, c0, c1)
    error_text = f'MSE: {errors["mse"]:.3f}\nRMSE: {errors["rmse"]:.3f}\nR²: {errors["r_squared"]:.3f}'
    plt.text(0.02, 0.98, error_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return plt


def run_experiment(n_points):
    """Run a single experiment with n_points"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {n_points} data points")
    print(f"{'='*60}")
    
    # Generate data
    true_c0, true_c1 = 2.5, 1.8  # True coefficients
    x, y, true_c0, true_c1 = generate_data(n_points, true_c0, true_c1)
    
    print(f"True coefficients: c0 = {true_c0:.3f}, c1 = {true_c1:.3f}")
    
    # Method 1: Manual calculation
    c0_manual, c1_manual = linear_regression_manual(x, y)
    errors_manual = calculate_errors(x, y, c0_manual, c1_manual)
    
    print(f"\nManual method:")
    print(f"  c0 = {c0_manual:.6f} (error: {abs(c0_manual - true_c0):.6f})")
    print(f"  c1 = {c1_manual:.6f} (error: {abs(c1_manual - true_c1):.6f})")
    print(f"  R² = {errors_manual['r_squared']:.6f}")
    
    # Method 2: Matrix method
    c0_matrix, c1_matrix = linear_regression_matrix(x, y)
    errors_matrix = calculate_errors(x, y, c0_matrix, c1_matrix)
    
    print(f"\nMatrix method:")
    print(f"  c0 = {c0_matrix:.6f} (error: {abs(c0_matrix - true_c0):.6f})")
    print(f"  c1 = {c1_matrix:.6f} (error: {abs(c1_matrix - true_c1):.6f})")
    print(f"  R² = {errors_matrix['r_squared']:.6f}")
    
    # Check if methods give same result
    if abs(c0_manual - c0_matrix) < 1e-10 and abs(c1_manual - c1_matrix) < 1e-10:
        print(f"\n✓ Both methods give identical results")
    else:
        print(f"\n⚠ Methods differ slightly (within tolerance)")
    
    # Plot results
    plt = plot_regression(x, y, c0_matrix, c1_matrix, true_c0, true_c1,
                         f"Linear Regression (n={n_points})")
    plt.savefig(f'regression_{n_points}_points.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'n_points': n_points,
        'true_c0': true_c0,
        'true_c1': true_c1,
        'c0': c0_matrix,
        'c1': c1_matrix,
        'c0_error': abs(c0_matrix - true_c0),
        'c1_error': abs(c1_matrix - true_c1),
        'r_squared': errors_matrix['r_squared']
    }

def run_multiple_experiments():
    """Run experiments with different numbers of points"""
    point_counts = [10, 50, 100, 500, 1000]
    results = []
    
    for n in point_counts:
        result = run_experiment(n)
        results.append(result)
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Coefficient error vs number of points
    ax1 = axes[0, 0]
    n_points = [r['n_points'] for r in results]
    c0_errors = [r['c0_error'] for r in results]
    c1_errors = [r['c1_error'] for r in results]
    
    ax1.plot(n_points, c0_errors, 'bo-', linewidth=2, markersize=8, label='c0 error')
    ax1.plot(n_points, c1_errors, 'ro-', linewidth=2, markersize=8, label='c1 error')
    ax1.set_xlabel('Number of points')
    ax1.set_ylabel('Absolute error')
    ax1.set_title('Coefficient Error vs Sample Size')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R-squared vs number of points
    ax2 = axes[0, 1]
    r_squared = [r['r_squared'] for r in results]
    ax2.plot(n_points, r_squared, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of points')
    ax2.set_ylabel('R-squared')
    ax2.set_title('Goodness of Fit vs Sample Size')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Estimated coefficients vs true values
    ax3 = axes[1, 0]
    c0_est = [r['c0'] for r in results]
    c1_est = [r['c1'] for r in results]
    true_c0 = results[0]['true_c0']
    true_c1 = results[0]['true_c1']
    
    width = 0.35
    x = np.arange(len(n_points))
    ax3.bar(x - width/2, c0_est, width, label=f'Estimated c0 (true={true_c0:.2f})')
    ax3.bar(x + width/2, c1_est, width, label=f'Estimated c1 (true={true_c1:.2f})')
    ax3.axhline(y=true_c0, color='blue', linestyle='--', alpha=0.5)
    ax3.axhline(y=true_c1, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of points')
    ax3.set_ylabel('Coefficient value')
    ax3.set_title('Estimated Coefficients vs True Values')
    ax3.set_xticks(x)
    ax3.set_xticklabels(n_points)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Convergence visualization
    ax4 = axes[1, 1]
    # Show last experiment (most points) for convergence check
    last_result = results[-1]
    # Regenerate data for visualization
    x_large, y_large, _, _ = generate_data(last_result['n_points'], 
                                          last_result['true_c0'], 
                                          last_result['true_c1'])
    
    # Plot with regression line
    ax4.scatter(x_large, y_large, alpha=0.3, s=10, label='Data points')
    x_line = np.linspace(min(x_large), max(x_large), 100)
    y_reg = last_result['c0'] + last_result['c1'] * x_line
    y_true = last_result['true_c0'] + last_result['true_c1'] * x_line
    ax4.plot(x_line, y_reg, 'r-', linewidth=3, label='Regression line')
    ax4.plot(x_line, y_true, 'g--', linewidth=2, label='True line')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(f'Convergence Check (n={last_result["n_points"]})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Linear Regression Analysis with Varying Sample Sizes', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('regression_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'n_points':<10} {'c0_est':<10} {'c0_error':<12} {'c1_est':<10} {'c1_error':<12} {'R²':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['n_points']:<10} {r['c0']:<10.4f} {r['c0_error']:<12.6f} "
              f"{r['c1']:<10.4f} {r['c1_error']:<12.6f} {r['r_squared']:<10.4f}")
    
    return results


def demonstrate_system_of_equations():
    """Demonstrate the SLE being solved"""
    print("\n" + "="*70)
    print("DEMONSTRATION: Solving the Normal Equations")
    print("="*70)
    
    # Generate small dataset for demonstration
    x, y, true_c0, true_c1 = generate_data(5, noise_level=0.2)
    
    print(f"Generated {len(x)} data points:")
    for i in range(len(x)):
        print(f"  Point {i+1}: x={x[i]:.3f}, y={y[i]:.3f}")
    
    print(f"\nTrue coefficients: c0 = {true_c0:.3f}, c1 = {true_c1:.3f}")
    
    # Calculate sums manually
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    print(f"\nCalculated sums:")
    print(f"  n = {n}")
    print(f"  Σx = {sum_x:.3f}")
    print(f"  Σy = {sum_y:.3f}")
    print(f"  Σx² = {sum_x2:.3f}")
    print(f"  Σxy = {sum_xy:.3f}")
    
    # Build normal equations
    print(f"\nNormal equations:")
    print(f"  [{n}, {sum_x:.3f}]   [c0]   [{sum_y:.3f}]")
    print(f"  [{sum_x:.3f}, {sum_x2:.3f}] * [c1] = [{sum_xy:.3f}]")
    
    # Solve manually
    ATA = np.array([[n, sum_x], [sum_x, sum_x2]])
    ATy = np.array([sum_y, sum_xy])
    
    det = ATA[0, 0] * ATA[1, 1] - ATA[0, 1] * ATA[1, 0]
    print(f"\nDeterminant: det = {det:.3f}")
    
    if det != 0:
        # Compute inverse
        ATA_inv = np.array([[ATA[1, 1], -ATA[0, 1]],
                            [-ATA[1, 0], ATA[0, 0]]]) / det
        
        # Solve
        c = ATA_inv @ ATy
        c0, c1 = c[0], c[1]
        
        print(f"\nSolution:")
        print(f"  [c0]   [{ATA_inv[0,0]:.3f} {ATA_inv[0,1]:.3f}]   [{sum_y:.3f}]   [{c0:.3f}]")
        print(f"  [c1] = [{ATA_inv[1,0]:.3f} {ATA_inv[1,1]:.3f}] * [{sum_xy:.3f}] = [{c1:.3f}]")
        
        print(f"\nFinal coefficients:")
        print(f"  c0 = {c0:.6f} (true: {true_c0:.6f}, error: {abs(c0 - true_c0):.6f})")
        print(f"  c1 = {c1:.6f} (true: {true_c1:.6f}, error: {abs(c1 - true_c1):.6f})")
        
        # Plot this small example
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color='blue', s=100, label='Data points')
        
        x_line = np.array([min(x), max(x)])
        y_reg = c0 + c1 * x_line
        y_true = true_c0 + true_c1 * x_line
        
        plt.plot(x_line, y_reg, 'r-', linewidth=2, label=f'Regression: y = {c0:.3f} + {c1:.3f}x')
        plt.plot(x_line, y_true, 'g--', linewidth=2, label=f'True: y = {true_c0:.3f} + {true_c1:.3f}x')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression Demonstration (n=5)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('regression_demo_small.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return c0, c1


def main():
    print("LINEAR REGRESSION IMPLEMENTATION")
    print("="*60)
    print("Solving y = c0 + c1*x using normal equations")
    print("="*60)
    
    # Demonstrate with small dataset
    c0_demo, c1_demo = demonstrate_system_of_equations()
    
    # Run experiments with different numbers of points
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS WITH VARYING SAMPLE SIZES")
    print("="*70)
    
    results = run_multiple_experiments()
    

if __name__ == "__main__":
    main()