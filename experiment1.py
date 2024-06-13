import sys
import numpy as np
from gauss_jacobi import gauss_jacobi


def experiment1():

    A = np.array([[5.0, 0.5, 0.5],
                  [8.0, 1.5, 5.0],
                  [7.0, 6.0, 5.0]])
    b = np.array([10.0, 15.0, 18.0])

    x0_list = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [20.0, 30.0, 40.0]]

    for x0 in x0_list:
        x, num_iter = gauss_jacobi(A, b, x0, max_iter=10 ** 5)
        print(f'Solução: {x}')
        print(f'Número de iterações: {num_iter}')

experiment1()