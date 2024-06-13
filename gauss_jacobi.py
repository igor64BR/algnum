import numpy as np
from numpy import zeros

def gauss_jacobi(A, b, x0, max_iter=1000, tol=1e-12):
    """
    Solves the linear system Ax = b using the Gauss-Jacobi iterative method.
    
    Parameters:
    A (numpy.ndarray): Coefficient matrix.
    b (numpy.ndarray): Constant vector.
    x0 (numpy.ndarray): Initial guess.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.
    
    Returns:
    x (numpy.ndarray): Solution vector.
    iterations (int): Number of iterations performed.
    """
    # Initialize variables
    x = np.copy(x0)
    n = len(b)
    iterations = 0
    
    # Iterate until max_iter
    for _ in range(max_iter):
        x_new = np.copy(x)
        
        # Update each element of x
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, iterations
        
        # Update x for the next iteration
        x = x_new
        iterations += 1
    
    # Return the last approximation if max_iter is reached without convergence
    return x, iterations

def gera_matriz_DIAGDOM(n): # Gera uma matriz de dimensão nxn com elementos aleatórios (entre 0 e 1) diagonalemnte dominante
    A = np.random.rand(n, n)
    for i in range(0, n):
            s = 0
            for j in range(0, n):
                s = s + A[i, j]  
            A[i, i] =  A[i, i] + s
    return (A)



def gera_b_para_Sistema_linear_com_solucao_unitaria(A, n):
    b = zeros(n, float)
    for i in range(0, n):
        s = 0
        for j in range(0, n):
            s = s + A[i, j]
        b[i] = s
    return b