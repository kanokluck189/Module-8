import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Change path if your csv is elsewhere
data = pd.read_csv("func_guess_data - func_guess_data.csv")

# assuming columns are named x and y
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# g(x) = c0*x + c1*x^2 + c2*sin(x) + c3*cos(x) + c4*e^x/100
A = np.column_stack([
    x,
    x**2,
    np.sin(x),
    np.cos(x),
    np.exp(x) / 100
])

# ---- LEAST SQUARES SOLUTION ----
coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)

# Round to closest integers (your guess)
guess = np.round(coeffs).astype(int)

print("Estimated coefficients:")
for i, c in enumerate(coeffs):
    print(f"c{i} = {c:.4f}")

print("\nRounded guess:")
for i, c in enumerate(guess):
    print(f"c{i} = {c}")

# ---- PLOT RESULT ----
x_plot = np.linspace(1, 10, 400)

y_plot = (
    coeffs[0]*x_plot +
    coeffs[1]*x_plot**2 +
    coeffs[2]*np.sin(x_plot) +
    coeffs[3]*np.cos(x_plot) +
    coeffs[4]*np.exp(x_plot)/100
)

plt.scatter(x, y)
plt.plot(x_plot, y_plot)
plt.title("Least Squares Approximation")
plt.show()
