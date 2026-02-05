import time

def generate_hilbert_matrix(n):
    # Generate Hilbert matrix of size n x n
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            H[i][j] = 1.0 / (i + j + 1)
    return H

def gaussian_elimination(A, b):
    # Solve Ax = b using Gaussian elimination
    n = len(b)
    
    # Make copies
    A = [row[:] for row in A]
    b = b[:]
    
    # Forward elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        max_val = abs(A[i][i])
        for k in range(i+1, n):
            if abs(A[k][i]) > max_val:
                max_val = abs(A[k][i])
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]
        
        # Check for singularity
        if abs(A[i][i]) < 1e-15:
            return None
        
        # Eliminate
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n-1, -1, -1):
        sum_ax = 0.0
        for j in range(i+1, n):
            sum_ax += A[i][j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i][i]
    
    return x

def vector_norm(v):
    #Compute L2 norm of a vector
    return sum(x*x for x in v) ** 0.5

def matrix_vector_multiply(A, x):
    #Multiply matrix A by vector x
    n = len(x)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * x[j]
    return result

def analyze_hilbert(n):
    # Analyze Hilbert matrix
    print(f"\nAnalyzing Hilbert matrix {n}x{n} :")
    
    # Generate matrix
    H = generate_hilbert_matrix(n)
    
    # Generate RHS for solution [1, 1, ..., 1]
    x_expected = [1.0] * n
    b = matrix_vector_multiply(H, x_expected)
    
    # Solve
    start = time.time()
    x_computed = gaussian_elimination(H, b)
    solve_time = time.time() - start
    
    if x_computed:
        # Compute error
        error = 0.0
        for i in range(n):
            error += (x_computed[i] - 1.0) ** 2
        error = error ** 0.5
        
        print(f"  Time: {solve_time:.4f} seconds")
        print(f"  Absolute error: {error:.6e}")
        
        # Print first few values
        print(f"  First 3 computed values:")
        for i in range(min(3, n)):
            print(f"    x[{i}] = {x_computed[i]:.8f}")
        
        return error
    else:
        print("  Failed to solve (matrix is singular)")
        return None

print("\n" + "="*60)
print("RESULTS")
print("="*60)

for n in [5, 10, 20, 50, 100]:
    analyze_hilbert(n)