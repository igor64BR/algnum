import random
import statistics

def produto_escalar(a, b):
    """
    Calcula o produto escalar de dois vetores a e b.
    """
    return sum(x * y for x, y in zip(a, b))

def gauss_jacobi(A, b, x0, tol=1e-12, max_iter=1000):
    """
    Implementa o método iterativo de Gauss-Jacobi para resolver um sistema linear Ax = b.

    Parâmetros:
    - A: lista de listas, matriz de coeficientes do sistema linear.
    - b: lista, vetor de termos independentes do sistema linear.
    - x0: lista, valores iniciais para a solução.
    - tol: float, tolerância para o critério de parada (padrão: 1e-12).
    - max_iter: int, número máximo de iterações (padrão: 1000).

    Retorna:
    - x: lista, vetor solução do sistema linear.
    - num_iter: int, número de iterações realizadas.
    """
    n = len(A)
    x = x0.copy()
    for k in range(max_iter):
        x_new = [0] * n
        for i in range(n):
            s1 = produto_escalar(A[i][:i], x[:i])
            s2 = produto_escalar(A[i][i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < tol:
            return x_new, k+1
        x = x_new
    return x, max_iter

# Experimento 1
A = [[10., -1., 2., 0.],
     [-1., 11., -1., 3.],
     [2., -1., 10., -1.],
     [0.0, 3., -1., 8.]]
b = [6., 25., -11., 15.]
x0_list = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
for x0 in x0_list:
    x, num_iter = gauss_jacobi(A, b, x0)
    print(f'Experimento 1:')
    print(f'Palpite inicial: {x0}')
    print(f'Solução: {x}, Número de iterações: {num_iter}')

# Experimento 2
def gerar_matriz_diagonalmente_dominante(n):
    A = [[random.random() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][i] += 15
    return A

def gerar_vetor(n):
    return [random.random() for _ in range(n)]

n = 3
A = gerar_matriz_diagonalmente_dominante(n)
b = gerar_vetor(n)
x0_list = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
num_iters = []
for x0 in x0_list:
    x, num_iter = gauss_jacobi(A, b, x0)
    num_iters.append(num_iter)
    print(f'\nExperimento 2:')
    print(f'Palpite inicial: {x0}')
    print(f'Solução: {x}, Número de iterações: {num_iter}')
print(f'Número médio de iterações: {statistics.mean(num_iters)}')

# Experimento 3
n = 20
A = gerar_matriz_diagonalmente_dominante(n)
b = gerar_vetor(n)
x0_list = [[0] * n, [1] * n, [2] * n, [3] * n]
num_iters = []
for x0 in x0_list:
    x, num_iter = gauss_jacobi(A, b, x0)
    num_iters.append(num_iter)
    print(f'\nExperimento 3:')
    print(f'Palpite inicial: {x0}')
    print(f'Solução: {x}, Número de iterações: {num_iter}')
print(f'Número médio de iterações: {statistics.mean(num_iters)}')
print(f'Desvio padrão das iterações: {statistics.stdev(num_iters)}')
