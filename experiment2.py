import numpy as np
from numpy import zeros
from algnum.igor_baiocoo import gauss_jacobi, gera_b_para_Sistema_linear_com_solucao_unitaria, gera_matriz_DIAGDOM

def experiment2():
    print("Experimento 1")
    np.random.seed(0)
    n = 3
    A = gera_matriz_DIAGDOM(n)
    b = gera_b_para_Sistema_linear_com_solucao_unitaria(A, n)
    x0_list = [np.random.uniform(0, 10, n) for _ in range(4)]  # Generate 4 distinct initial guesses
    print("A: [")

    [print(e) for e in A]

    print("]")
    for i, x0 in enumerate(x0_list):
        x, num_iter = gauss_jacobi(A, b, x0)
        print(f'\nDimensão: {n}')
        print(f'Chute {i+1}: {x0}')
        print(f'Solução: {x}, Número de iterações: {num_iter}')
    print()

experiment2()
