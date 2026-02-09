import matplotlib.pyplot as plt
import numpy as np
import random

# 1. Define interesting functions
def f1(x):
    return np.sin(x) * np.cos(2*x)

def f2(x):
    return np.exp(-x) * np.sin(5*x)

def f3(x):
    return np.abs(x) * np.sin(x)

def f4(x):
    return 1 / (1 + 25*x**2)  # Runge's phenomenon

def f5(x):
    return np.tanh(3*x) * np.cos(4*x)

# 2. Interpolation function
def interpolate(f, a, b, n):
    # Create n+1 equally spaced points
    x_points = np.linspace(a, b, n+1)
    y_points = f(x_points)
    
    # Build Vandermonde matrix for polynomial coefficients
    # A * c = y, where A[i,j] = x_points[i]**j
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            A[i, j] = x_points[i]**j
    
    # Solve for coefficients c
    c = np.linalg.solve(A, y_points)
    
    # Create fine grid for plotting polynomial
    x_fine = np.linspace(a, b, 1000)
    y_poly = np.zeros_like(x_fine)
    
    # Evaluate polynomial: c0 + c1*x + c2*x^2 + ... + cn*x^n
    for i in range(n+1):
        y_poly += c[i] * (x_fine**i)
    
    return x_points, y_points, c, x_fine, y_poly

# 3. Plotting function
def plot_interpolation(f, a, b, n, func_name, ax):
    x_points, y_points, c, x_fine, y_poly = interpolate(f, a, b, n)
    y_true = f(x_fine)
    
    # Plot
    ax.plot(x_fine, y_true, 'b-', linewidth=2, alpha=0.5, label='True function')
    ax.plot(x_fine, y_poly, 'r-', linewidth=2, label=f'Degree {n} polynomial')
    ax.scatter(x_points, y_points, color='green', s=50, zorder=5, label='Interpolation points')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{func_name} on [{a},{b}], n={n}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add polynomial equation in title or text box
    poly_eq = f'p(x) = {c[0]:.2f}'
    for i in range(1, min(4, len(c))):
        poly_eq += f' + {c[i]:.2f}x^{i}'
    if len(c) > 4:
        poly_eq += ' + ...'
    
    ax.text(0.05, 0.95, poly_eq, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Test different functions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Test cases
test_cases = [
    (f1, -np.pi, np.pi, 5, 'sin(x)cos(2x)', axes[0,0]),
    (f2, 0, 3, 6, 'exp(-x)sin(5x)', axes[0,1]),
    (f3, -3, 3, 7, '|x|sin(x)', axes[0,2]),
    (f4, -1, 1, 8, '1/(1+25x²)', axes[1,0]),
    (f5, -2, 2, 10, 'tanh(3x)cos(4x)', axes[1,1])
]

for f, a, b, n, name, ax in test_cases:
    plot_interpolation(f, a, b, n, name, ax)

# 6. Add one more interesting case with higher degree
ax6 = axes[1,2]
def f6(x):
    return np.sin(x**2) + 0.1*x

plot_interpolation(f6, -2, 2, 12, 'sin(x²)+0.1x', ax6)

plt.tight_layout()
plt.savefig('polynomial_interpolation.png', dpi=150, bbox_inches='tight')
plt.show()

# 7. Show Runge's phenomenon with increasing degrees
print("\nRunge's Phenomenon Demonstration (1/(1+25x²)):")
print("="*50)

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
degrees = [5, 10, 15, 20]

for i, n in enumerate(degrees):
    ax = axes2[i//2, i%2]
    x_points, y_points, c, x_fine, y_poly = interpolate(f4, -1, 1, n)
    y_true = f4(x_fine)
    
    ax.plot(x_fine, y_true, 'b-', linewidth=2, alpha=0.5, label='True function')
    ax.plot(x_fine, y_poly, 'r-', linewidth=2, label=f'Degree {n}')
    ax.scatter(x_points, y_points, color='green', s=30, zorder=5)
    
    # Calculate error
    error = np.max(np.abs(y_poly - y_true))
    ax.set_title(f'n={n}, Max error={error:.3f}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.5)
    ax.legend()

plt.suptitle("Runge's Phenomenon: Oscillations increase with higher degree", fontsize=12)
plt.tight_layout()
plt.savefig('runge_phenomenon.png', dpi=150, bbox_inches='tight')
plt.show()

# 8. Print coefficients for one example
print("\nCoefficients for f(x) = sin(x)cos(2x), n=5:")
print("="*50)
x_points, y_points, c, x_fine, y_poly = interpolate(f1, -np.pi, np.pi, 5)
for i, coeff in enumerate(c):
    print(f"c{i} = {coeff:.6f}")
print(f"\nPolynomial: p(x) = {c[0]:.3f} + {c[1]:.3f}x + {c[2]:.3f}x² + {c[3]:.3f}x³ + {c[4]:.3f}x⁴ + {c[5]:.3f}x⁵")