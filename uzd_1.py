import numpy as np


def mprint(A):
    with np.printoptions(precision=3, suppress=True):
        print(A)
        print()


def gauss_jordan(A):
    mprint(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio = A[j][i] / A[i][i]
                A[j][i:] = A[j][i:] - ratio * A[i][i:]
        mprint(A)

    # make 1's in diagonal and save result
    x = np.zeros(n)
    for i in range(n):
        x[i] = A[i][n] / A[i][i]
    return x


# 1, 3, 1, 20
A = np.array(
    [
        [4, 1, 1, 7],
        [1, 0, 2, -2],
        [2, 2, -7, 1],
        [4, 14, 7, 0]
    ]).astype(np.float)
b = np.array([148, -37, 21, 53]).astype(np.float).reshape(-1, 1)

A1 = np.hstack((A, b))
print(gauss_jordan(A1))


