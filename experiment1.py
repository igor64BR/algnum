import numpy as np
from gauss_jacobi import gauss_jacobi


def experiment1():
    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0.0, 3., -1., 8.]])
    b = np.array([6., 25., -11., 15.])
    x0 = np.zeros(4)
    x, num_iter = gauss_jacobi(A, b, x0)
    print(f'Solução: {x}, Número de iterações: {num_iter}')

experiment1()