import numpy as np
import matplotlib.pyplot as plt

# Points
x_points = [0, 1, 2]
y_points = [1, 2, 0]

# Solve for spline coefficients
# We have 8 unknowns: a0, b0, c0, d0, a1, b1, c1, d1

# Set up system of equations
A = np.zeros((8, 8))
b = np.zeros(8)

# 1. S0(0) = 1
A[0, 0] = 1  # a0
b[0] = 1

# 2. S0(1) = 2
A[1, 0] = 1  # a0
A[1, 1] = 1  # b0
A[1, 2] = 1  # c0
A[1, 3] = 1  # d0
b[1] = 2

# 3. S1(1) = 2  
A[2, 4] = 1  # a1
A[2, 5] = 1  # b1
A[2, 6] = 1  # c1
A[2, 7] = 1  # d1
b[2] = 2

# 4. S1(2) = 0
A[3, 4] = 1  # a1
A[3, 5] = 2  # 2*b1
A[3, 6] = 4  # 4*c1
A[3, 7] = 8  # 8*d1
b[3] = 0

# 5. First derivative continuity at x=1: S0'(1) = S1'(1)
# S0'(x) = b0 + 2c0*x + 3d0*x^2
# S1'(x) = b1 + 2c1*x + 3d1*x^2
A[4, 1] = 1   # b0
A[4, 2] = 2   # 2c0
A[4, 3] = 3   # 3d0
A[4, 5] = -1  # -b1
A[4, 6] = -2  # -2c1
A[4, 7] = -3  # -3d1
b[4] = 0

# 6. Second derivative continuity at x=1: S0''(1) = S1''(1)
# S0''(x) = 2c0 + 6d0*x
# S1''(x) = 2c1 + 6d1*x
A[5, 2] = 2   # 2c0
A[5, 3] = 6   # 6d0
A[5, 6] = -2  # -2c1
A[5, 7] = -6  # -6d1
b[5] = 0

# 7. Natural boundary: S0''(0) = 0
# S0''(0) = 2c0
A[6, 2] = 2  # 2c0
b[6] = 0

# 8. Natural boundary: S1''(2) = 0
# S1''(2) = 2c1 + 12d1
A[7, 6] = 2   # 2c1
A[7, 7] = 12  # 12d1
b[7] = 0

# Solve the system
coeffs = np.linalg.solve(A, b)

# Extract coefficients
a0, b0, c0, d0, a1, b1, c1, d1 = coeffs

print("Natural Cubic Spline Coefficients:")
print("="*40)
print(f"Segment 1 (x ∈ [0,1]):")
print(f"  a0 = {a0:.6f}")
print(f"  b0 = {b0:.6f}")
print(f"  c0 = {c0:.6f}")
print(f"  d0 = {d0:.6f}")
print(f"  S0(x) = {a0:.3f} + {b0:.3f}x + {c0:.3f}x² + {d0:.3f}x³")

print(f"\nSegment 2 (x ∈ [1,2]):")
print(f"  a1 = {a1:.6f}")
print(f"  b1 = {b1:.6f}")
print(f"  c1 = {c1:.6f}")
print(f"  d1 = {d1:.6f}")
print(f"  S1(x) = {a1:.3f} + {b1:.3f}x + {c1:.3f}x² + {d1:.3f}x³")

# Verify conditions
print("\nVerification:")
print("="*40)
print(f"S0(0) = {a0:.6f} (should be 1.0)")
print(f"S0(1) = {a0 + b0 + c0 + d0:.6f} (should be 2.0)")
print(f"S1(1) = {a1 + b1 + c1 + d1:.6f} (should be 2.0)")
print(f"S1(2) = {a1 + 2*b1 + 4*c1 + 8*d1:.6f} (should be 0.0)")
print(f"\nS0'(1) = {b0 + 2*c0 + 3*d0:.6f}")
print(f"S1'(1) = {b1 + 2*c1 + 3*d1:.6f}")
print(f"S0''(1) = {2*c0 + 6*d0:.6f}")
print(f"S1''(1) = {2*c1 + 6*d1:.6f}")
print(f"S0''(0) = {2*c0:.6f} (should be 0.0)")
print(f"S1''(2) = {2*c1 + 12*d1:.6f} (should be 0.0)")

# Create spline functions
def S0(x):
    return a0 + b0*x + c0*x**2 + d0*x**3

def S1(x):
    return a1 + b1*x + c1*x**2 + d1*x**3

# Plot the spline
x_fine1 = np.linspace(0, 1, 100)
x_fine2 = np.linspace(1, 2, 100)

plt.figure(figsize=(10, 6))

# Plot spline segments
plt.plot(x_fine1, S0(x_fine1), 'b-', linewidth=2, label='S₀(x), x ∈ [0,1]')
plt.plot(x_fine2, S1(x_fine2), 'r-', linewidth=2, label='S₁(x), x ∈ [1,2]')

# Plot data points
plt.scatter(x_points, y_points, color='green', s=100, zorder=5, label='Data points')

# Mark the connection point
plt.scatter([1], [S0(1)], color='orange', s=80, zorder=5, label='Connection point')

# Add vertical line at connection
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

# Add tangent line at connection point to show smoothness
slope = b0 + 2*c0 + 3*d0
tangent_x = np.linspace(0.7, 1.3, 50)
tangent_y = S0(1) + slope*(tangent_x - 1)
plt.plot(tangent_x, tangent_y, 'g--', alpha=0.7, linewidth=1.5, label='Tangent at x=1')

plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Natural Cubic Spline Interpolation\nPoints: (0,1), (1,2), (2,0)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# Add text with polynomial equations
plt.text(0.05, 0.95, f'S₀(x) = {a0:.3f} + {b0:.3f}x + {c0:.3f}x² + {d0:.3f}x³', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.text(0.05, 0.88, f'S₁(x) = {a1:.3f} + {b1:.3f}x + {c1:.3f}x² + {d1:.3f}x³', 
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

plt.tight_layout()
plt.savefig('natural_cubic_spline.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional plot showing derivatives to verify smoothness
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Function values
ax1 = axes[0]
x_fine = np.linspace(0, 2, 200)
y_fine = np.piecewise(x_fine, 
                     [x_fine <= 1, x_fine > 1], 
                     [lambda x: S0(x), lambda x: S1(x)])
ax1.plot(x_fine, y_fine, 'b-', linewidth=2)
ax1.scatter(x_points, y_points, color='red', s=80, zorder=5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Cubic Spline: Function')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

# First derivative
ax2 = axes[1]
def dS0(x):
    return b0 + 2*c0*x + 3*d0*x**2

def dS1(x):
    return b1 + 2*c1*x + 3*d1*x**2

d_fine = np.piecewise(x_fine, 
                     [x_fine <= 1, x_fine > 1], 
                     [lambda x: dS0(x), lambda x: dS1(x)])
ax2.plot(x_fine, d_fine, 'g-', linewidth=2)
ax2.scatter([1], [dS0(1)], color='red', s=80, zorder=5)
ax2.set_xlabel('x')
ax2.set_ylabel("S'(x)")
ax2.set_title('First Derivative (Continuous)')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

# Second derivative
ax3 = axes[2]
def ddS0(x):
    return 2*c0 + 6*d0*x

def ddS1(x):
    return 2*c1 + 6*d1*x

dd_fine = np.piecewise(x_fine, 
                      [x_fine <= 1, x_fine > 1], 
                      [lambda x: ddS0(x), lambda x: ddS1(x)])
ax3.plot(x_fine, dd_fine, 'r-', linewidth=2)
ax3.scatter([1], [ddS0(1)], color='red', s=80, zorder=5)
ax3.scatter([0], [ddS0(0)], color='orange', s=80, zorder=5, label='S₀''(0)=0')
ax3.scatter([2], [ddS1(2)], color='orange', s=80, zorder=5, label='S₁''(2)=0')
ax3.set_xlabel('x')
ax3.set_ylabel("S''(x)")
ax3.set_title('Second Derivative (Continuous, Natural Boundaries)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Smoothness Verification of Natural Cubic Spline', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('spline_smoothness_verification.png', dpi=150, bbox_inches='tight')
plt.show()