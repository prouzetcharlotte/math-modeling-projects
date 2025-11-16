import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import time


def generate_corrected_sparse_tridiagonal_matrix(n, diagonal_value=5, off_diagonal_value=1):
    """
    Generates a sparse tridiagonal matrix, ensuring no overlaps.

    Args:
        n: Dimension of the system (size of the matrix A).
        diagonal_value: Value for the diagonal elements.
        off_diagonal_value: Value for the off-diagonal elements.

    Returns:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
    """
    ### TODO: Complete code here 
    # Main diagonal
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n-1, off_diagonal_value)

    # Construct sparse matrix
    data = np.concatenate([main_diag,off_diag,off_diag])
    print(data)
    rows = np.concatenate([np.arange(n), np.arange(n-1), np.arange(1,n)]) # might need to use np.concatenate
    print(rows)
    cols = np.concatenate([np.arange(n), np.arange(1,n), np.arange(n-1)])
    print(cols)
    As = csr_matrix((data, (rows, cols)), shape=(n, n))
    print(As)

    #Construct dense matrix
    A_dense=np.zeros((n,n))
    A_dense[n-1,n-1]=diagonal_value
    for i in range(n-1):
        A_dense[i, i] = diagonal_value
        A_dense[i,i+1]=off_diagonal_value
        A_dense[i+1,i]=off_diagonal_value

    b = np.random.rand(n)
    return As, A_dense, b



def jacobi_dense(A, b, x0, tol=1e-6, max_iter=1000):
    """
    Jacobi method for dense matrices.

    Args:
        A: Dense coefficient matrix (numpy array).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        x: Approximate solution vector.
        iterations: Number of iterations performed.
        time_taken: Time taken for the iterations.
    """
    ### TODO: Code your thing here!
    start_time=time.time()
    n = A.shape[0]
    x = x0.copy()
    errors = []
    for k in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            x_new[i]=(1/A[i,i])*(b[i]-np.dot(A[i,:],x)+A[i,i]*x[i])
        error = np.linalg.norm(x_new-x)
        errors.append(error)
        x = x_new
        if error < tol:
            break
    end_time=time.time()
    time_taken=end_time-start_time
    return x, k + 1, time_taken

def jacobi_sparse(A, b, x0, tol=1e-7, max_iter=10000):
    start_time=time.time()
    x=x0.copy()
    D1=1/A.diagonal()
    LU=A-sparse.diags(A.diagonal())
    for k in range(max_iter):
        x_new = D1*(b-LU.dot(x))
        error = np.linalg.norm(x_new-x)
        if error < tol:
            break
        x = x_new
    end_time=time.time()
    time_taken=end_time-start_time
    return x, k + 1, time_taken


# Example usage:
n=1000
x0 = np.zeros(n)  ## initial guess
A_sparse, A_dense_v1, b = generate_corrected_sparse_tridiagonal_matrix(n) 
A_dense_v2 = A_sparse.toarray()  # Convert to dense format for comparison


# Classical Jacobi (dense)
x_dense, iter_dense, time_dense = jacobi_dense(A_dense_v2, b, x0) 

# Jacobi for sparse matrix
x_sparse, iter_sparse, time_sparse = jacobi_sparse(A_sparse, b, x0)

print(f"Iterations (dense): {iter_dense}, Time (dense): {time_dense:.4f} seconds") #0.4183 seconds
print(f"Iterations (sparse): {iter_sparse}, Time (sparse): {time_sparse:.4f} seconds") #0.0015 seconds


x_exact = np.linalg.solve(A_dense_v2, b)
print(x_exact)
print(x_sparse)


### TODO: 
# Implement a small for loop comparing the times required for both approaches as a function of the dimension n

