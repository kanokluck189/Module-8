import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    # Combine A and b into an augmented matrix
    # We use float64 to avoid integer division issues
    M = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    #Forward Elimination
    for i in range(n):
        # 1. Partial Pivoting: Find the largest element in this column
        max_row = i + np.argmax(np.abs(M[i:, i]))
        M[[i, max_row]] = M[[max_row, i]]
        
        if abs(M[i, i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular!")

        #Make all rows below this one 0 in the current column
        for j in range(i + 1, n):
            ratio = M[j, i] / M[i, i]
            M[j, i:] -= ratio * M[i, i:]

    #Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, n] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    
    return x

#Test Case

A = np.array([[2, 1, -1], 
              [-3, -1, 2], 
              [-2, 1, 2]])
b = np.array([8, -11, -3])

try:
    solution = gaussian_elimination(A, b)
    print("Matrix A:\n", A)
    print("\nVector b:", b)
    print("\nSolution [x, y, z]:")
    print(solution)
    
    # Verification
    print("\nVerification (A * solution):")
    print(np.dot(A, solution))
except ValueError as e:
    print(e)