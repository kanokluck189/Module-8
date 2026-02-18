import numpy as np
from typing import Callable


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 1000
):
    """
    Universal Gradient Descent optimizer.

    Parameters:
        f        : function to minimize
        grad     : gradient of the function
        x0       : initial point (np.ndarray of any dimension)
        lr       : learning rate
        tol      : stopping tolerance
        max_iter : maximum iterations
    """
    x = x0.astype(float)
    history = []

    for i in range(max_iter):
        g = grad(x)
        history.append((i, x.copy(), f(x)))

        x_new = x - lr * g

        # stopping condition
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            history.append((i + 1, x.copy(), f(x)))
            break

        x = x_new

    return x, history


########################################
# Function: (x+1)^2 + (y-1)^2 + (z-2)^2
# Written in a dimension-independent way
########################################

TARGET = np.array([-1, 1, 2])  # change this vector for other dimensions


def f(v: np.ndarray) -> float:
    return np.sum((v - TARGET) ** 2)


def grad_f(v: np.ndarray) -> np.ndarray:
    return 2 * (v - TARGET)


########################################
# Run the algorithm
########################################

if __name__ == "__main__":

    # Initial point (same dimension as TARGET)
    x0 = np.array([10.0, -5.0, 0.0])

    solution, history = gradient_descent(
        f=f,
        grad=grad_f,
        x0=x0,
        lr=0.2
    )

    print("First 5 iterations:\n")
    for h in history[:5]:
        print(f"Iter {h[0]} | x = {h[1]} | f(x) = {h[2]:.6f}")

    print("\nFinal Result:")
    print("Solution:", solution)
    print("Function value:", f(solution))
    print("Iterations:", len(history))
