import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# Matrix Generator
# =====================================================

def hilbert_matrix(n, dtype=np.float64):
    return np.array([[1/(i+j+1) for j in range(n)] for i in range(n)], dtype=dtype)


# =====================================================
# Solvers
# =====================================================

def gaussian_solve(A, b):
    return np.linalg.solve(A, b)


def qr_solve(A, b):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ b)


def svd_solve(A, b):
    U, s, Vt = np.linalg.svd(A)
    S_inv = np.diag(1/s)
    return Vt.T @ S_inv @ U.T @ b


SOLVERS = {
    "Gaussian": gaussian_solve,
    "QR": qr_solve,
    "SVD": svd_solve
}


# =====================================================
# Experiment
# =====================================================

def run_experiment():

    sizes = [5, 8, 10, 12, 15]
    dtype = np.float64
    noise_level = 0.0

    plt.figure()

    print("\n====== Numerical Stability Experiment ======\n")

    for solver_name, solver_func in SOLVERS.items():

        errors = []

        print(f"\n--- {solver_name} ---\n")

        for n in sizes:

            A = hilbert_matrix(n, dtype=dtype)
            x_true = np.ones(n, dtype=dtype)
            b = A @ x_true

            # optional noise
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, size=n).astype(dtype)
                b += noise

            x_computed = solver_func(A, b)

            relative_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
            condition_number = np.linalg.cond(A)

            errors.append(relative_error)

            print(f"Size: {n}")
            print(f"Condition number: {condition_number:.2e}")
            print(f"Relative error: {relative_error:.2e}\n")

        plt.plot(sizes, errors, marker='o', label=solver_name)

    plt.yscale('log')
    plt.xlabel("Matrix Size")
    plt.ylabel("Relative Error (log)")
    plt.title("Algorithm Stability Comparison (Hilbert Matrix)")
    plt.legend()
    plt.grid()
    plt.show()


# =====================================================

if __name__ == "__main__":
    run_experiment()
