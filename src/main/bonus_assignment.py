import numpy as np
np.set_printoptions(precision=7, suppress=True, linewidth=100)

# Question 1
# Set up and run gauss seidel method
def gauss_seidel(gauss_mat, gauss_aug, initial_guess, tol, num_iterations):
    x = initial_guess.copy()
    n = len(x)
    for j in range(num_iterations):
        x_new = np.zeros(n)
        for i in range(n):
            k1 = np.dot(gauss_mat[i, :i], x_new[:i])
            k2 = np.dot(gauss_mat[i, i + 1:], x[i + 1:])
            x_new[i] = (gauss_aug[i] - k1 - k2) / gauss_mat[i, i]
        if np.linalg.norm(x - x_new, np.inf) < tol:
            return j + 1
        x = x_new.copy()
    return 
  
# Question 2
# Set up and run jacobi method
def jacobi_method(mat, aug, initial_guess, tol, num_iterations):
    diagonalized_mat = np.diag(np.diag(mat))
    LU = mat - diagonalized_mat
    x = initial_guess
    for i in range(1, num_iterations + 1):
        x_new = np.linalg.inv(diagonalized_mat).dot(aug - LU.dot(x))
        if np.linalg.norm(x_new - x) < tol:
            return i
        x = x_new
    return num_iterations

# Question 3
# Set up f(x), derivative, and newton raphson method and run
def f(x):
    return x**3 - x**2 + 2
def df(x):
    return 3*x**2 - 2*x 
def newton_raphson(newt: float):
    tol = 1e-6
    new_nw = newt
    res = f(newt) / df(newt)
    nw_ct = 1
    while abs(res) >= tol:
        new_nw = new_nw - res
        
        nw_ct = 1 + nw_ct
        
        res = f(new_nw) / df(new_nw)
    return nw_ct

# Question 4
# Set up and run divided difference method
def divided_difference(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i + 2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            lc: float = matrix[i][j - 1]
            dlc: float = matrix[i - 1][j - 1]
            num: float = (lc - dlc)
            den = matrix[i][0] - matrix[i - j + 1][0]
            ops = num / den
            matrix[i][j] = ops
    return matrix
# Set up and run hermite poly function
def hermite_poly():
    herm_x = [0, 1, 2]
    herm_fx = [1, 2, 4]
    herm_prime = [1.06, 1.23, 1.55]
    m = len(herm_x)
    matrix = np.zeros((2 * m, 2 * m))    
    for i, x in enumerate(herm_x):
        matrix[2 * i][0] = x
        matrix[2 * i + 1][0] = x   
    for i, y in enumerate(herm_fx):
        matrix[2 * i][1] = y
        matrix[2 * i + 1][1] = y
    for i, f_prime in enumerate(herm_prime):
        matrix[2 * i + 1][2] = f_prime
    finalQ4 = divided_difference(matrix)
    print(finalQ4, end="\n\n")

# Question 5
# Set up (x, y) function and run
def function(x, y):
    return y - x**3
# Set up modified euler method function and run
def modified_euler_method(initial_point, start_of_t, end_of_t, num_of_iterations):
    h = (end_of_t - start_of_t) / num_of_iterations
    t = start_of_t
    y = initial_point
    for cur_iteration in range(num_of_iterations):
        k1 = function(t, y)
        k2 = function(t+h, y+h*k1)
        y += h/2 * (k1 + k2)
        t += h
    return y    
# Set up 'if' function and run
if __name__ == "__main__":
  gauss_mat = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
  aug_mat = np.array([1, 3, 0])
  initial_guess = np.array([0, 0, 0])
  num_iterations = 50
  tol = 1e-6
  print(gauss_seidel(gauss_mat, aug_mat, initial_guess, tol, num_iterations), end = "\n\n")
  mat = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
  aug = np.array([1, 3, 0])
  initial_guess = np.array([0, 0, 0])
  tol = 1e-6
  num_iterations = 50
  print(jacobi_method(mat, aug, initial_guess, tol, num_iterations), end = "\n\n")
  newt = 0.5
  print(newton_raphson(newt), end="\n\n")
  hermite_poly()
  initial_point = 0.5
  start_of_t = 0
  end_of_t = 3
  num_of_iterations = 100 
  print("%.5f" %modified_euler_method(initial_point, start_of_t, end_of_t, num_of_iterations))
  print()
