import numpy as np


def test_plane(x):
    return 2 * x[0] - x[1] + 3*x[2] + x[3] - x[4]


n = 1000
m = 5

a = -1
b = 1

# random points on the plane
x_vals = np.random.uniform(a, b, size=(m, n))

y_vals_exact = test_plane(x_vals)

noise = np.random.normal(0, 0.5, size=y_vals_exact.shape)
y = y_vals_exact + noise

X = np.stack(x_vals, axis=1)

c = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

max_iters = 10000
alpha = 0.0000008
eps = 0.000001

for i in range(max_iters):
    c -= alpha * X.T @ (X @ c - y)
    loss = np.linalg.norm(X @ c - y) ** 2

    step = alpha * X.T @ (X @ c - y)
    if np.linalg.norm(step) < eps:
        break

    print(f"{i}: loss: {loss:.2f}; c = {c}")