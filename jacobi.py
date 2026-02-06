import random
import time

def make_diag_dom(n, factor=1.5):
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = 0.0
        for j in range(n):
            if i != j:
                A[i][j] = random.uniform(-10, 10)
                row_sum += abs(A[i][j])
        A[i][i] = row_sum * factor
        if random.random() > 0.5:
            A[i][i] = -A[i][i]
    return A

def make_solution(n):
    return [random.uniform(-5, 5) for _ in range(n)]

def calc_b(A, x):
    n = len(A)
    b = [0.0 for _ in range(n)]
    for i in range(n):
        for j in range(n):
            b[i] += A[i][j] * x[j]
    return b

def jacobi_solve(A, b, max_iter=1000, tol=1e-8):
    n = len(A)
    x_old = [0.0 for _ in range(n)]
    x_new = [0.0 for _ in range(n)]
    
    for it in range(1, max_iter + 1):
        for i in range(n):
            s = 0.0
            for j in range(n):
                if i != j:
                    s += A[i][j] * x_old[j]
            x_new[i] = (b[i] - s) / A[i][i]
        
        err = 0.0
        for i in range(n):
            diff = abs(x_new[i] - x_old[i])
            if diff > err:
                err = diff
        
        if err < tol:
            return x_new, it, True
        
        x_old = x_new[:]
    
    return x_new, max_iter, False

def calc_error(x1, x2):
    n = len(x1)
    max_err = 0.0
    for i in range(n):
        diff = abs(x1[i] - x2[i])
        if diff > max_err:
            max_err = diff
    return max_err

def test_size(n, factor=2.0):
    print(f"\nTesting {n}x{n}, dominance factor {factor}")
    
    A = make_diag_dom(n, factor)
    x_true = make_solution(n)
    b = calc_b(A, x_true)
    
    start = time.time()
    x_calc, it, conv = jacobi_solve(A, b)
    t = time.time() - start
    
    if conv:
        err = calc_error(x_calc, x_true)
        print(f"  Converged in {it} it, time={t:.4f}s, error={err:.2e}")
        return True
    else:
        print(f"  Failed to converge in {it} iterations")
        return False

def main():
    print("Jacobi Method Test")
    print("=" * 40)
    
    # Test different sizes
    sizes = [5, 10, 20, 50]
    for n in sizes:
        test_size(n)
    
    # Test weak dominance
    print("\n\nWeak dominance test (factor=1.05):")
    test_size(10, 1.05)
    
    # Test strong dominance
    print("\n\nStrong dominance test (factor=5.0):")
    test_size(10, 5.0)
    
    # Test non-dominant (should fail)
    print("\n\nNon-dominant matrix test (should fail):")
    n = 5
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = 1.0
            else:
                A[i][j] = 10.0
    
    x_true = make_solution(n)
    b = calc_b(A, x_true)
    
    x_calc, it, conv = jacobi_solve(A, b, max_iter=100)
    
    if conv:
        print("  Unexpected: converged!")
    else:
        print(f"  Expected: failed to converge in {it} it")
    
    # Show some solution values
    print("\n\nExample solution (5x5):")
    A = make_diag_dom(5)
    x_true = make_solution(5)
    b = calc_b(A, x_true)
    
    x_calc, it, conv = jacobi_solve(A, b)
    
    if conv:
        print("True solution (first 3):", [f"{v:.4f}" for v in x_true[:3]])
        print("Jacobi solution (first 3):", [f"{v:.4f}" for v in x_calc[:3]])
        err = calc_error(x_calc, x_true)
        print(f"Max error: {err:.2e}")

if __name__ == "__main__":
    main()