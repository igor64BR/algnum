from gauss_jacobi import gauss_jacobi

def experiment1():
    A = [[4, -1, 0], [1, 3, -1], [2, -1, 5]]
    b = [1, 2, -1]
    initial_guesses = [[0, 0, 0], [1, 1, 1], [-1, -1, -1], [2, 2, 2]]

    for i, x_0 in enumerate(initial_guesses):
        solution, iterations = gauss_jacobi(A, b, x_0)
        print(f"Solution {i+1}: {solution}")
        print(f"Iterations: {iterations}\n")
