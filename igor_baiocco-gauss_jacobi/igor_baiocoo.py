import random
import numpy as np
from numpy import zeros

# Aluno: Igor Baiocco Rodrigues


def gauss_jacobi(A, b, x0, max_iter=1000, tol=1e-12):
    """
    Resolve o sistema linear Ax = b usando o método iterativo de Gauss-Jacobi.
    
    Parâmetros:
    A (numpy.ndarray): Matriz de coeficientes.
    b (numpy.ndarray): Vetor constante.
    x0 (numpy.ndarray): Estimativa inicial.
    max_iter (int): Número máximo de iterações.
    tol (float): Tolerância para convergência.
    
    Retorna:
    x (numpy.ndarray): Vetor solução.
    iterations (int): Número de iterações realizadas.
    """
    # Inicializa as variáveis
    x = np.copy(x0)
    n = len(b)
    iterations = 0
    
    # Itera até max_iter
    for _ in range(max_iter):
        x_new = np.copy(x)
        
        # Atualiza cada elemento de x
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_new[i] = (b[i] - s) / A[i][i]
        
        # Verifica a convergência
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new, iterations
        
        # Atualiza x para a próxima iteração
        x = x_new
        iterations += 1
    
    # Retorna a última aproximação se max_iter for atingido sem convergência
    return x, iterations

def gera_matriz_DIAGDOM(n): # Gera uma matriz de dimensão nxn com elementos aleatórios (entre 0 e 1) diagonalmente dominante
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

def experiment1():
    print("Experimento 1")
    A = np.array([[5.0, 0.5, 0.5],
                  [8.0, 1.5, 5.0],
                  [7.0, 6.0, 5.0]])
    b = np.array([10.0, 15.0, 18.0])

    x0_list = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [20.0, 30.0, 40.0], [2.0, 3.0, 4.0]]

    SPACER = setSpacer()
    print(SPACER)
    for x0 in x0_list:
        x, num_iter = gauss_jacobi(A, b, x0, max_iter=1000)
        print(f'Solução: {x}')
        print(f'Número de iterações: {num_iter}')
        print(SPACER)
    print()

def experiment2(n=3, experiment=2):
    print("Experimento", experiment)
    np.random.seed(random.randint(0, 1000)) # linha adicionada para gerar novos resultados
    A = gera_matriz_DIAGDOM(n)
    b = gera_b_para_Sistema_linear_com_solucao_unitaria(A, n)
    x0_list = [np.random.uniform(0, 10, n) for _ in range(4)]  # Gera 4 estimativas iniciais distintas
    print("A: [")

    [print(e) for e in A]

    print("]")
    SPACER = setSpacer()
    print(SPACER)
    for i, x0 in enumerate(x0_list):
        x, num_iter = gauss_jacobi(A, b, x0)
        print(f'\nDimensão: {n}')
        print(f'Chute {i+1}: {x0}')
        print(f'Solução: {x}, Número de iterações: {num_iter}')
        print(SPACER)
    print()

def experiment3():
    experiment2(n=20, experiment=3)

def setSpacer():
    """
    Esta função cria uma string de espaçamento consistindo de uma barra inclinada para a frente, seguida por 100 traços e terminando com uma barra inclinada para a frente.
    """
    SPACER = '/'
    for _ in range(100):
        SPACER += '-'
    SPACER += '/'
    return SPACER

SPACER = setSpacer()

print(SPACER)
experiment1()
print(SPACER)
experiment2()
print(SPACER)
experiment3()
print(SPACER)
