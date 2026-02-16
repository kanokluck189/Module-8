import numpy as np
import matplotlib.pyplot as plt

# Locator coordinates from the image
locators = np.array([
    [5, 10],
    [50, 20],
    [100, 10],
    [150, 30],
    [130, 70],
    [120, 120],
    [60, 130],
    [30, 110],
    [5, 60],
    [2, 120]
], dtype=float)

# Approx distances
dists = np.array([
    90.9,
    73.7,
    108,
    133,
    102,
    89.3,
    48.5,
    13.0,
    45.3,
    40.1
], dtype=float)

# Multilateration via linearized least squares
x1, y1 = locators[0]
r1 = dists[0]

A = []
b = []

for (xi, yi), ri in zip(locators[1:], dists[1:]):
    A.append([2*(xi - x1), 2*(yi - y1)])
    b.append(r1**2 - ri**2 - x1**2 + xi**2 - y1**2 + yi**2)

A = np.array(A)
b = np.array(b)

# Solve least squares
position, *_ = np.linalg.lstsq(A, b, rcond=None)
x_est, y_est = position

print("Estimated Battleship Coordinates:", (x_est, y_est))

# Plot
plt.figure()
plt.scatter(locators[:,0], locators[:,1], label="Locators")
plt.scatter(x_est, y_est, marker='x', s=200, label="Battleship")
for i, (x,y) in enumerate(locators):
    plt.text(x+1, y+1, str(i+1))

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Multilateration Result")
plt.show()
