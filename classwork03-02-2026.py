from math import sqrt

def g(x):
    return sqrt(x + 2)

x = 3  # starting point

for k in range(20):
    x = g(x)
    print(f"Iteration {k+1}: xk = {x}")
