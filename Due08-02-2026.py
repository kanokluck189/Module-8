import numpy as np
import matplotlib.pyplot as plt

class SolverSuite:
    def __init__(self, f, df, g, a, b, x0, tol=1e-10, max_iter=50):
        self.f = f    # Function
        self.df = df  # Derivative (for Newton)
        self.g = g    # Fixed point form x = g(x)
        self.a, self.b = a, b # Interval (Bisection)
        self.x0 = x0  # Starting point (Newton/Fixed)
        self.tol = tol
        self.max_iter = max_iter

    def run_bisection(self):
        errors = []
        a, b = self.a, self.b
        for _ in range(self.max_iter):
            err = abs(b - a)
            errors.append(err)
            if err < self.tol: break
            mid = (a + b) / 2
            if self.f(a) * self.f(mid) < 0: b = mid
            else: a = mid
        return errors

    def run_newton(self):
        errors = []
        x = self.x0
        for _ in range(self.max_iter):
            fx = self.f(x)
            dfx = self.df(x)
            if abs(dfx) < 1e-12: break
            x_new = x - fx / dfx
            err = abs(x_new - x)
            errors.append(err)
            x = x_new
            if err < self.tol: break
        return errors

    def run_fixed_point(self):
        errors = []
        x = self.x0
        for _ in range(self.max_iter):
            x_new = self.g(x)
            err = abs(x_new - x)
            errors.append(err)
            x = x_new
            if err < self.tol: break
        return errors

def plot_experiment(title, suite):
    plt.figure(figsize=(10, 6))
    
    # Run algorithms
    bis_err = suite.run_bisection()
    newt_err = suite.run_newton()
    fp_err = suite.run_fixed_point()
    
    plt.semilogy(bis_err, 'b-o', label='Bisection', markersize=4)
    plt.semilogy(newt_err, 'r-s', label='Newton', markersize=4)
    plt.semilogy(fp_err, 'g-^', label='Fixed Point', markersize=4)
    
    plt.title(f"Convergence Comparison: {title}")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error (log scale)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()

# EXPERIMENTS CONFIGURATION

# Base Case: x^2 - 4 = 0
ex1 = SolverSuite(
    f = lambda x: x**2 - 4,
    df = lambda x: 2*x,
    g = lambda x: x - 0.1*(x**2 - 4), # alpha = 0.1
    a=0, b=5, x0=5
)

# Cubic (Multiple roots/slow convergence): (x-1)^3 = 0
# Newton is slower here because the derivative is zero at the root!
ex2 = SolverSuite(
    f = lambda x: (x-1)**3,
    df = lambda x: 3*(x-1)**2,
    g = lambda x: x - 0.2*(x-1)**3,
    a=0, b=3, x0=2.5
)

# Trig/Exponential Mix: cos(x) - x = 0
ex3 = SolverSuite(
    f = lambda x: np.cos(x) - x,
    df = lambda x: -np.sin(x) - 1,
    g = lambda x: np.cos(x),
    a=0, b=1, x0=0.5
)

# Run plots
plot_experiment("Simple Quadratic (x^2 - 4)", ex1)
plot_experiment("Cubic (x-1)^3 - Multiple Root Case", ex2)
plot_experiment("Trigonometric (cos(x) - x)", ex3)