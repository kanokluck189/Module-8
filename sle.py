import numpy as np
import matplotlib.pyplot as plt
import time
import random

# ==================== MATRIX GENERATORS ====================

def generate_random_matrix(n, min_val=-5, max_val=5, diag_dom=False, dom_factor=1.5):
    """Generate random matrix"""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = random.uniform(min_val, max_val)
    
    if diag_dom:
        for i in range(n):
            row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
            A[i, i] = row_sum * dom_factor + 0.1
    return A

def generate_hilbert_matrix(n):
    """Generate Hilbert matrix"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H

# ==================== SOLVERS ====================

def gaussian_elimination(A, b):
    """Gaussian elimination"""
    n = len(b)
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    
    for i in range(n):
        # Pivot
        max_row = i
        max_val = abs(A[i, i])
        for k in range(i+1, n):
            if abs(A[k, i]) > max_val:
                max_val = abs(A[k, i])
                max_row = k
        
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        if abs(A[i, i]) < 1e-15:
            return np.zeros(n), False
        
        # Eliminate
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Back sub
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
    return x, True

def jacobi(A, b, tol=1e-6, max_iter=5000):
    """Jacobi iteration"""
    n = len(b)
    x = np.zeros(n)
    x_new = np.zeros(n)
    
    for it in range(1, max_iter+1):
        for i in range(n):
            s = 0
            for j in range(n):
                if i != j:
                    s += A[i, j] * x[j]
            x_new[i] = (b[i] - s) / A[i, i]
        
        if np.linalg.norm(x_new - x) < tol:
            return x_new, it, True
        
        x = x_new.copy()
    
    return x_new, max_iter, False

def gauss_seidel(A, b, tol=1e-6, max_iter=5000):
    """Gauss-Seidel iteration"""
    n = len(b)
    x = np.zeros(n)
    
    for it in range(1, max_iter+1):
        x_old = x.copy()
        
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s) / A[i, i]
        
        if np.linalg.norm(x - x_old) < tol:
            return x, it, True
    
    return x, max_iter, False

# ==================== EXPERIMENTS ====================

def run_experiments():
    """Run all experiments and print results"""
    print("="*85)
    print(f"{'Problem type':<15} {'Algorithm':<10} {'Dimension':<10} {'Iterations':<10} {'Residue':<15} {'Abs. error':<15}")
    print("="*85)
    
    results = []
    
    # Experiment 1: Random matrix (no diagonal dominance)
    for n in [3, 5, 10, 20, 30, 40, 50]:
        A = generate_random_matrix(n, diag_dom=False)
        x_true = np.ones(n)
        b = np.dot(A, x_true)
        
        x_ge, success = gaussian_elimination(A, b)
        if success:
            residue = np.linalg.norm(np.dot(A, x_ge) - b)
            error = np.linalg.norm(x_ge - x_true)
            results.append(['RandMat', 'Gauss', n, 1, residue, error])
            print(f"{'RandMat':<15} {'Gauss':<10} {n:<10} {1:<10} {residue:<15.2e} {error:<15.2e}")
    
    print("-"*85)
    
    # Experiment 2: Hilbert matrix
    for n in [3, 5, 8, 12, 16, 20, 25]:
        A = generate_hilbert_matrix(n)
        x_true = np.ones(n)
        b = np.dot(A, x_true)
        
        # Gaussian
        x_ge, success = gaussian_elimination(A, b)
        if success:
            residue = np.linalg.norm(np.dot(A, x_ge) - b)
            error = np.linalg.norm(x_ge - x_true)
            results.append(['Hilbert', 'Gauss', n, 1, residue, error])
            print(f"{'Hilbert':<15} {'Gauss':<10} {n:<10} {1:<10} {residue:<15.2e} {error:<15.2e}")
        
        # Gauss-Seidel
        x_gs, it_gs, conv = gauss_seidel(A, b, tol=1e-6, max_iter=2000)
        if conv:
            residue = np.linalg.norm(np.dot(A, x_gs) - b)
            error = np.linalg.norm(x_gs - x_true)
            results.append(['Hilbert', 'Seidel', n, it_gs, residue, error])
            print(f"{'Hilbert':<15} {'Seidel':<10} {n:<10} {it_gs:<10} {residue:<15.2e} {error:<15.2e}")
        else:
            print(f"{'Hilbert':<15} {'Seidel':<10} {n:<10} {'FAIL':<10} {'-':<15} {'-':<15}")
    
    print("-"*85)
    
    # Experiment 3: Diagonally dominant matrix
    for n in [3, 5, 10, 15, 20, 25, 30]:
        A = generate_random_matrix(n, diag_dom=True, dom_factor=1.5)
        x_true = np.ones(n)
        b = np.dot(A, x_true)
        
        # Gaussian
        x_ge, success = gaussian_elimination(A, b)
        if success:
            residue = np.linalg.norm(np.dot(A, x_ge) - b)
            error = np.linalg.norm(x_ge - x_true)
            results.append(['DiagDominant', 'Gauss', n, 1, residue, error])
            print(f"{'DiagDominant':<15} {'Gauss':<10} {n:<10} {1:<10} {residue:<15.2e} {error:<15.2e}")
        
        # Jacobi
        x_j, it_j, conv_j = jacobi(A, b, tol=1e-6, max_iter=3000)
        if conv_j:
            residue = np.linalg.norm(np.dot(A, x_j) - b)
            error = np.linalg.norm(x_j - x_true)
            results.append(['DiagDominant', 'Jacobi', n, it_j, residue, error])
            print(f"{'DiagDominant':<15} {'Jacobi':<10} {n:<10} {it_j:<10} {residue:<15.2e} {error:<15.2e}")
        else:
            print(f"{'DiagDominant':<15} {'Jacobi':<10} {n:<10} {'FAIL':<10} {'-':<15} {'-':<15}")
        
        # Gauss-Seidel
        x_gs, it_gs, conv_gs = gauss_seidel(A, b, tol=1e-6, max_iter=3000)
        if conv_gs:
            residue = np.linalg.norm(np.dot(A, x_gs) - b)
            error = np.linalg.norm(x_gs - x_true)
            results.append(['DiagDominant', 'Seidel', n, it_gs, residue, error])
            print(f"{'DiagDominant':<15} {'Seidel':<10} {n:<10} {it_gs:<10} {residue:<15.2e} {error:<15.2e}")
        else:
            print(f"{'DiagDominant':<15} {'Seidel':<10} {n:<10} {'FAIL':<10} {'-':<15} {'-':<15}")
        
        print("-"*85)
    
    return results

def plot_results(results):
    """Plot the results"""
    # Convert to numpy array for easier processing
    results_array = np.array(results, dtype=object)
    
    # Extract data for each problem type
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Random Matrix - Gaussian error
    ax = axes[0, 0]
    rand_data = results_array[results_array[:, 0] == 'RandMat']
    n_vals = rand_data[:, 2].astype(float)
    errors = rand_data[:, 5].astype(float)
    ax.plot(n_vals, errors, 'bo-', linewidth=2)
    ax.set_xlabel('Matrix Size (n)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Random Matrix - Gaussian Elimination')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Hilbert Matrix comparison
    ax = axes[0, 1]
    hilbert_data = results_array[results_array[:, 0] == 'Hilbert']
    for method in ['Gauss', 'Seidel']:
        method_data = hilbert_data[hilbert_data[:, 1] == method]
        if len(method_data) > 0:
            n_vals = method_data[:, 2].astype(float)
            errors = method_data[:, 5].astype(float)
            ax.plot(n_vals, errors, 'o-', linewidth=2, label=method)
    ax.set_xlabel('Matrix Size (n)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Hilbert Matrix - Gauss vs Seidel')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Diagonal Dominant - Iterations
    ax = axes[1, 0]
    diag_data = results_array[results_array[:, 0] == 'DiagDominant']
    for method in ['Jacobi', 'Seidel']:
        method_data = diag_data[diag_data[:, 1] == method]
        if len(method_data) > 0:
            n_vals = method_data[:, 2].astype(float)
            iters = method_data[:, 3].astype(float)
            ax.plot(n_vals, iters, 'o-', linewidth=2, label=method)
    ax.set_xlabel('Matrix Size (n)')
    ax.set_ylabel('Iterations')
    ax.set_title('Diagonal Dominant - Iterations Needed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: All methods error comparison for n=20
    ax = axes[1, 1]
    n_val = 20
    methods_data = {}
    for prob_type in ['RandMat', 'Hilbert', 'DiagDominant']:
        type_data = results_array[results_array[:, 0] == prob_type]
        n_data = type_data[type_data[:, 2].astype(float) == n_val]
        for row in n_data:
            method = row[1]
            error = float(row[5])
            if method not in methods_data:
                methods_data[method] = []
            methods_data[method].append((prob_type, error))
    
    # Plot grouped bars
    methods = ['Gauss', 'Jacobi', 'Seidel']
    prob_types = ['RandMat', 'Hilbert', 'DiagDominant']
    bar_width = 0.25
    x = np.arange(len(prob_types))
    
    for idx, method in enumerate(methods):
        errors = []
        for prob in prob_types:
            found = False
            for m, err_list in methods_data.items():
                if m == method:
                    for prob_t, err in err_list:
                        if prob_t == prob:
                            errors.append(err)
                            found = True
                            break
                if found:
                    break
            if not found:
                errors.append(0)
        
        if any(err > 0 for err in errors):
            ax.bar(x + idx*bar_width, errors, bar_width, label=method)
    
    ax.set_xlabel('Problem Type')
    ax.set_ylabel('Absolute Error (log scale)')
    ax.set_title(f'Error Comparison for n={n_val}')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(prob_types)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sle_experiment_results.png', dpi=150)
    plt.show()
    
    # Additional plot: Error vs Matrix Size for all
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    colors = {'Gauss': 'red', 'Jacobi': 'blue', 'Seidel': 'green'}
    markers = {'RandMat': 'o', 'Hilbert': 's', 'DiagDominant': '^'}
    
    for row in results:
        prob_type, method, n, iters, residue, error = row
        n = float(n)
        error = float(error)
        
        if error > 0:
            ax2.scatter(n, error, color=colors.get(method, 'black'), 
                       marker=markers.get(prob_type, 'o'), s=50, 
                       label=f'{prob_type}-{method}' if prob_type == 'DiagDominant' and method == 'Gauss' else "")
    
    ax2.set_xlabel('Matrix Size (n)')
    ax2.set_ylabel('Absolute Error (log scale)')
    ax2.set_title('All Experiments: Error vs Matrix Size')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['DiagDominant-Gauss'], loc='upper left')
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Gauss'),
        Patch(facecolor='blue', label='Jacobi'),
        Patch(facecolor='green', label='Seidel'),
        Patch(facecolor='white', edgecolor='black', label='○ RandMat'),
        Patch(facecolor='white', edgecolor='black', label='□ Hilbert'),
        Patch(facecolor='white', edgecolor='black', label='△ DiagDominant')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('sle_all_results.png', dpi=150)
    plt.show()

def save_to_csv(results):
    """Save results to CSV file"""
    with open('sle_experiment_results.csv', 'w') as f:
        f.write("Problem_type,Algorithm,Dimension,Iterations,Residue,Abs_error\n")
        for row in results:
            prob_type, method, n, iters, residue, error = row
            f.write(f"{prob_type},{method},{n},{iters},{residue:.6e},{error:.6e}\n")
    
    print("\nResults saved to 'sle_experiment_results.csv'")

# ==================== MAIN ====================

def main():
    print("SYSTEM OF LINEAR EQUATIONS ALGORITHMS TESTING")
    print("="*85)
    print("Testing Gaussian Elimination, Jacobi, and Gauss-Seidel methods")
    print("on Random, Hilbert, and Diagonally Dominant matrices.")
    print("="*85)
    
    # Run experiments
    results = run_experiments()
    
    # Plot results
    plot_results(results)
    
    # Save to CSV
    save_to_csv(results)
    
    # Print summary statistics
    print("\n" + "="*85)
    print("SUMMARY STATISTICS")
    print("="*85)
    
    # Calculate averages
    gauss_errors = []
    jacobi_iters = []
    seidel_iters = []
    
    for row in results:
        prob_type, method, n, iters, residue, error = row
        error = float(error)
        iters = int(iters)
        
        if method == 'Gauss' and error > 0:
            gauss_errors.append(error)
        elif method == 'Jacobi' and iters > 0:
            jacobi_iters.append(iters)
        elif method == 'Seidel' and iters > 0:
            seidel_iters.append(iters)
    
    if gauss_errors:
        print(f"Gaussian Elimination - Average error: {np.mean(gauss_errors):.2e}")
        print(f"                     - Max error: {np.max(gauss_errors):.2e}")
    
    if jacobi_iters:
        print(f"Jacobi Method       - Average iterations: {np.mean(jacobi_iters):.0f}")
        print(f"                     - Max iterations: {np.max(jacobi_iters)}")
    
    if seidel_iters:
        print(f"Gauss-Seidel Method - Average iterations: {np.mean(seidel_iters):.0f}")
        print(f"                     - Max iterations: {np.max(seidel_iters)}")
    
    # Compare convergence speed
    if jacobi_iters and seidel_iters:
        speedup = np.mean(jacobi_iters) / np.mean(seidel_iters)
        print(f"\nGauss-Seidel is {speedup:.1f}x faster than Jacobi on average")

if __name__ == "__main__":
    main()