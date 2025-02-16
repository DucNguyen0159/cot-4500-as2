import numpy as np

def neville():
    """
    Question 1: Neville’s method
    """
    x = np.array([3.6, 3.8, 3.9], dtype=float)
    f = np.array([1.675, 1.436, 1.318], dtype=float)
    w = 3.7
    n = len(x)

    # Initialize the table
    neville_table = np.zeros((n, n))
    for i in range(n):
        neville_table[i, 0] = f[i]

    # Fill the table
    for i in range(1, n):
        for j in range(1, i + 1):
            term1 = (w - x[i - j]) * neville_table[i, j - 1]
            term2 = (w - x[i]) * neville_table[i - 1, j - 1]
            denominator = x[i] - x[i - j]
            neville_table[i, j] = (term1 - term2) / denominator

    # Print the second degree interpolating value for f(3.7)
    print(neville_table[n - 1, n - 1], "\n")


def newton():
    """
    Question 2, 3: Newton’s forward method
    """
    x = np.array([7.2, 7.4, 7.5, 7.6], dtype=float)
    f = np.array([23.5492, 25.3913, 26.8224, 27.4589], dtype=float)
    n = len(x)

    # Create difference table
    dif = np.zeros((n, n))
    for i in range(n):
        dif[i, 0] = f[i]

    # Compute divided differences
    for i in range(1, n):
        for j in range(1, i + 1):
            dif[i, j] = (dif[i, j - 1] - dif[i - 1, j - 1]) / (x[i] - x[i - j])

    # Print the diagonal (the Newton coefficients)
    for i in range(1, n):
        print(dif[i, i])
    print()

    # Question 3: approximate f(7.3)
    w = 7.3
    p1 = dif[0, 0] + dif[1, 1] * (w - x[0])
    p2 = p1 + dif[2, 2] * (w - x[0]) * (w - x[1])
    p3 = p2 + dif[3, 3] * (w - x[0]) * (w - x[1]) * (w - x[2])
    print(p3, "\n")


def hermite():
    x = [3.6, 3.6, 3.8, 3.8, 3.9, 3.9]
    fx = [1.675, 1.675, 1.436, 1.436, 1.318, 1.318]
    fpx = [-1.195, -1.195, -1.188, -1.188, -1.182, -1.182]
    
    n = len(x)
    table = [[0] * (n - 1) for _ in range(n)] 

    # Initialize first column = x
    for i in range(n):
        table[i][0] = x[i]
        table[i][1] = fx[i]  # Second column = f(x)

    # Compute f'
    for i in range(1, n):
        if x[i] == x[i - 1]:
            table[i][2] = fpx[i]  # Use f'(x) if same x value
        else:
            table[i][2] = (table[i][1] - table[i - 1][1]) / (x[i] - x[i - 1])

    # Compute higher-order divided differences
    for j in range(3, n - 1):
        for i in range(j - 1, n):
            table[i][j] = (table[i][j - 1] - table[i - 1][j - 1]) / (x[i] - x[i - j + 1])

    # Print the Hermite Divided Difference Table
    for i in range(n):
        print("[ ", end="")
        for j in range(n - 1): 
            print(f"{table[i][j]: 12.8e} ", end="")
        print("]")
    
    print()


def cubic_spline():
    """
    Question 5: Cubic spline
    """
    x = np.array([2, 5, 8, 10], dtype=float)
    f = np.array([3, 5, 7, 9], dtype=float)
    n = len(x)

    # Compute h 
    h = np.zeros(n - 1, dtype=float)
    for i in range(n - 1):
        h[i] = x[i + 1] - x[i]

    # Construct matrix A
    A = np.zeros((n, n), dtype=float)
    A[0, 0] = 1.0
    A[n - 1, n - 1] = 1.0

    # Fill A
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i]     = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    # Construct b vector
    b = np.zeros(n, dtype=float)
    for i in range(1, n - 1):
        term1 = (3.0 / h[i]) * (f[i + 1] - f[i])
        term2 = (3.0 / h[i - 1]) * (f[i] - f[i - 1])
        b[i] = term1 - term2

    # Solve for c in A*c = b
    c = np.linalg.solve(A, b)

    # Print matrix A
    for row in A:
        print("[", end="")
        for val in row:
            print(f"{val:3.0f}.", end="")
        print("]")

    # Print vector b
    print("[", end="")
    for val in b:
        print(f"{val:2.0f}.", end="")
    print("]")

    # Print vector c
    print("[0.", end="")
    for val in c[1:-1]:     # Loop over the middle elements
        print(f" {val:.8f}", end="")
    print(" 0.]\n")


def main():
    # Question 1: Neville’s method
    neville()

    # Question 2, 3: Newton’s forward method
    newton()

    # Question 4: Hermite polynomial approximation matrix
    hermite()

    # Question 5: Cubic spline
    cubic_spline()


if __name__ == "__main__":
    main()

