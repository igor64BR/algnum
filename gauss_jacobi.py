import numpy as np

def gauss_jacobi(A, b, x0, tol=1e-12, max_iter=1000):
    """
    Implementa o método iterativo de Gauss-Jacobi para resolver um sistema linear Ax = b.

    Parâmetros:
    - A: numpy.ndarray, matriz de coeficientes do sistema linear.
    - b: numpy.ndarray, vetor de termos independentes do sistema linear.
    - x0: numpy.ndarray, vetor de valores iniciais para a solução.
    - tol: float, tolerância para critério de parada (default: 1e-12).
    - max_iter: int, número máximo de iterações (default: 1000).

    Retorna:
    - x: numpy.ndarray, vetor solução do sistema linear.
    - num_iter: int, número de iterações realizadas.
    """
    n = A.shape[0]
    x = x0.copy()
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.max(np.abs(x_new - x)) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter