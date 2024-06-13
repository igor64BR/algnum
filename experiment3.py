import numpy as np
from gauss_jacobi import gauss_jacobi


def experiment3():
    np.random.seed(0)
    n = 20
    A = np.random.rand(n, n) + np.diag([15]*n)  # Matriz com diagonal dominante
    b = np.random.rand(n)
    x0 = np.random.rand(n)
    x, num_iter = gauss_jacobi(A, b, x0)
    print(f'\nDimensão: {n}')
    print(f'Solução: {x}, Número de iterações: {num_iter}')

experiment3()