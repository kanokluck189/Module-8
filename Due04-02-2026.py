from math import fabs

# Function and derivative
def f(x):
    return x**2 - x - 2  # root at x=2

def df(x):
    return 2*x - 1


# ---------------- Bisection ----------------
def bisection(a, b, eps):
    print("\n=== Bisection ===")
    i = 0
    while (b - a)/2 > eps:
        c = (a + b) / 2
        i += 1
        print(f"Iter {i}: x = {c}, f(x) = {f(c)}")
        
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c
    return (a + b) / 2


# ---------------- Newton ----------------
def newton(x0, eps):
    print("\n=== Newton ===")
    i = 0
    x = x0
    while fabs(f(x)) > eps:
        x = x - f(x)/df(x)
        i += 1
        print(f"Iter {i}: x = {x}, f(x) = {f(x)}")
    return x


# ---------------- Relaxation ----------------
def relaxation(x0, alpha, eps):
    print(f"\n=== Relaxation (alpha={alpha}) ===")
    i = 0
    x = x0
    while fabs(f(x)) > eps:
        x = x + alpha * f(x)   # g(x) = x + alpha*f(x)
        i += 1
        print(f"Iter {i}: x = {x}, f(x) = {f(x)}")
    return x


# Interval and starting points
a, b = 1, 4
x0 = 3

# ---- Experiments ----
epsilons = [1e-2, 1e-6]

for eps in epsilons:
    print("\n\n#############################")
    print(f"Running with epsilon = {eps}")
    print("#############################")
    
    root_b = bisection(a, b, eps)
    print("Bisection root:", root_b)
    
    root_n = newton(x0, eps)
    print("Newton root:", root_n)
    
    # try two alphas
    root_r1 = relaxation(x0=2.5, alpha=-0.2, eps=eps)
    print("Relaxation root:", root_r1)
    
    root_r2 = relaxation(x0=2.5, alpha=-0.4, eps=eps)
    print("Relaxation root:", root_r2)
