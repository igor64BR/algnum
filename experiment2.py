import numpy as np
from numpy import zeros
from gauss_jacobi import gauss_jacobi, gera_matriz_DIAGDOM

def gera_b_para_Sistema_linear_com_solucao_unitaria(A, n):
    b = zeros(n, float)
    for i in range(0, n):
        s = 0
        for j in range(0, n):
            s = s + A[i, j]
        b[i] = s
    return b

def experiment2():
    np.random.seed(0)
    n = 3
    A = gera_matriz_DIAGDOM(n)
    b = gera_b_para_Sistema_linear_com_solucao_unitaria(A, n)  # Update this line
    x0 = np.random.rand(n)
    x, num_iter = gauss_jacobi(A, b, x0)
    print(f'\nDimensão: {n}')
    print(f'Solução: {x}, Número de iterações: {num_iter}')

experiment2()
