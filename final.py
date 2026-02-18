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
    "1": ("Gaussian Elimination", gaussian_solve),
    "2": ("QR Decomposition", qr_solve),
    "3": ("SVD (Most Stable)", svd_solve)
}


# =====================================================
# Experiment Function
# =====================================================

def run_experiment(sizes, solver_func, dtype, noise_level):

    errors = []
    conditions = []

    print("\nRunning experiment...\n")

    for n in sizes:

        A = hilbert_matrix(n, dtype=dtype)

        x_true = np.ones(n, dtype=dtype)
        b = A @ x_true

        # Add optional noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=n).astype(dtype)
            b = b + noise

        # Solve
        x_computed = solver_func(A, b)

        # Metrics
        relative_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
        condition_number = np.linalg.cond(A)

        errors.append(relative_error)
        conditions.append(condition_number)

        print(f"Matrix size: {n}")
        print(f"Condition number: {condition_number:.2e}")
        print(f"Relative error: {relative_error:.2e}\n")

    return errors, conditions


# =====================================================
# Visualization
# =====================================================

def plot_results(sizes, errors, conditions, solver_name):

    # Error explosion
    plt.figure()
    plt.plot(sizes, errors, marker='o')
    plt.yscale('log')
    plt.xlabel("Matrix Size")
    plt.ylabel("Relative Error (log scale)")
    plt.title(f"Error Explosion — {solver_name}")
    plt.grid()
    plt.show()

    # Condition vs Error
    plt.figure()
    plt.plot(conditions, errors, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Condition Number (log)")
    plt.ylabel("Relative Error (log)")
    plt.title(f"Error vs Ill-Conditioning — {solver_name}")
    plt.grid()
    plt.show()


# =====================================================
# Compare ALL Algorithms (Professor Favorite)
# =====================================================

def compare_algorithms(sizes, dtype, noise_level):

    plt.figure()

    for key in SOLVERS:

        solver_name, solver_func = SOLVERS[key]

        errors, _ = run_experiment(
            sizes,
            solver_func,
            dtype,
            noise_level
        )

        plt.plot(sizes, errors, marker='o', label=solver_name)

    plt.yscale('log')
    plt.xlabel("Matrix Size")
    plt.ylabel("Relative Error (log)")
    plt.title("Algorithm Stability Comparison")
    plt.legend()
    plt.grid()
    plt.show()


# =====================================================
# CLI Interface
# =====================================================

def main():

    print("\n====== Numerical Stability Lab ======\n")

    # ---------- Matrix sizes ----------
    sizes_input = input(
        "Enter matrix sizes separated by commas (default: 5,8,10,12,15): "
    )

    if sizes_input.strip() == "":
        sizes = [5, 8, 10, 12, 15]
    else:
        sizes = [int(x) for x in sizes_input.split(",")]

    # ---------- Precision ----------
    print("\nChoose precision:")
    print("1 - float64 (recommended)")
    print("2 - float32 (less stable)")

    precision_choice = input("Enter choice: ")

    dtype = np.float64 if precision_choice != "2" else np.float32

    # ---------- Noise ----------
    noise_input = input(
        "\nEnter noise level (example 1e-10, default = 0): "
    )

    noise_level = float(noise_input) if noise_input.strip() != "" else 0.0

    # ---------- Algorithm ----------
    print("\nChoose algorithm:")
    for key, value in SOLVERS.items():
        print(f"{key} - {value[0]}")

    print("4 - Compare ALL (recommended)")

    choice = input("Enter choice: ")

    # ---------- Run ----------
    if choice == "4":
        compare_algorithms(sizes, dtype, noise_level)

    else:
        solver_name, solver_func = SOLVERS.get(choice, SOLVERS["1"])

        errors, conditions = run_experiment(
            sizes,
            solver_func,
            dtype,
            noise_level
        )

        plot_results(sizes, errors, conditions, solver_name)


# =====================================================

if __name__ == "__main__":
    main()
