import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import time
def generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4):
    main_diag = np.full(n, diagonal_value)
    off_diag = np.full(n-1, off_diagonal_value)

    # Construct sparse matrix
    data = np.concatenate([main_diag,off_diag,off_diag])
    rows = np.concatenate([np.arange(n), np.arange(n-1), np.arange(1,n)]) 
    cols = np.concatenate([np.arange(n), np.arange(1,n), np.arange(n-1)])
    As = csr_matrix((data, (rows, cols)), shape=(n, n)) 

    # Construct dense matrix for reference
    A_dense = np.zeros((n, n))
    A_dense[n-1,n-1]=diagonal_value
    for i in range(n-1):
        A_dense[i, i] = diagonal_value
        A_dense[i,i+1]=off_diagonal_value
        A_dense[i+1,i]=off_diagonal_value
    b = np.random.rand(n)
    return As, A_dense, b

def successive_over_relaxation(A,b,x0, x_exact, tol=1e-5, max_iter=1000,w=1.2):
    x=x0.copy()
    D=sparse.diags(A.diagonal())
    L=sparse.tril(A,k=-1)
    U=sparse.triu(A,k=1)
    C=(1/w)*(D+w*L)
    errors = []
    C1=sparse.linalg.inv(C)
    for k in range(max_iter):
        x_new = C1.dot(b-(((w-1)/w)*D+U).dot(x))
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, k + 1, errors

def jacobi_sparse_with_error(A, b, x0, x_exact, tol=1e-5, max_iter=1000):
    ### TODO: 
    """
    Args:
        A: Sparse coefficient matrix (scipy.sparse.csr_matrix).
        b: Right-hand side vector (numpy array).
        x0: Initial guess for the solution vector (numpy array).
        x_exact : The true solution, used to compute ||x^(k)-x_exact||
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
    Returns:
        x : Approximate solution vector.
        iterations: Number of iterations performed.
        errors: List of errors between exact and approximate solution at each iteration.
    """
    x=x0.copy()
    D1=1/A.diagonal()
    LU=A-sparse.diags(A.diagonal())
    errors = []
    for k in range(max_iter):
        x_new = D1*(b-LU.dot(x))
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, k + 1, errors

def gauss_seidel_sparse_with_error(A, b, x0, x_exact, tol=1e-5, max_iter=1000):
    x=x0.copy()
    C=sparse.tril(A)  #correspond à D-L
    U=sparse.triu(A)-sparse.diags(A.diagonal())
    errors = []
    C1=sparse.linalg.inv(C)
    for k in range(max_iter):
        x_new = C1*(b-U.dot(x))
        error = np.linalg.norm(x_new-x_exact)
        errors.append(error)
        if error < tol:
            break
        x = x_new
    return x, k + 1, errors

def jacobi_dense(A, b, x0, tol=1e-5, max_iter=1000):
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
    return x, k + 1, errors

def plot_error(errors, iterations, errors2, iterations2,errors3,iterations3,errors4,iterations4):
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(iterations), errors, marker='o', linestyle='-')  # Use semilogy for log-scale on y-axis
    plt.semilogy(range(iterations2), errors2, marker='o', linestyle='-')
    plt.semilogy(range(iterations3), errors3, marker='o', linestyle='-')
    plt.semilogy(range(iterations4), errors4, marker='o', linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Error estimate")
    plt.title("Error vs Iterations")
    plt.grid(True)
    plt.show()

n=100
x0=np.zeros(n)
A, A_dense, b=generate_simple_sparse_tridiagonal_matrix(n, diagonal_value=10, off_diagonal_value=4)
x_exact=sparse.linalg.inv(A)*b
x, iter, errors=successive_over_relaxation(A,b,x0, x_exact)
x_j, iter_j,errors_j=jacobi_sparse_with_error(A, b, x0, x_exact)
x_jd,iter_jd,errors_jd=jacobi_dense(A_dense, b, x0)
x_gs, iter_gs, errors_gs=gauss_seidel_sparse_with_error(A, b, x0, x_exact)

