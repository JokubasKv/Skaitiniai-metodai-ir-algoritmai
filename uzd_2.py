import numpy as np
from numpy import sin, cos, arccos, pi
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')


def chebyshev_polynomial(x, i):
    return cos(i * arccos(x))


def to_chebyshev_interval(x, a, b):
    return (2 * x) / (b - a) - (b + a) / (b - a)


def interpolate_chebyshev(x, y, n):
    x_chebyshev = to_chebyshev_interval(x, 0, 11)
    A = chebyshev_polynomial(x_chebyshev, x.T)
    coefficients = np.linalg.solve(A, y)

    x_new = np.linspace(0, 11, 1000).reshape(-1, 1)
    xc = to_chebyshev_interval(x_new, 0, 11)

    A = chebyshev_polynomial(xc, x.T)
    y_new = A.dot(coefficients)

    plt.title('Spain temperatures 2012')
    plt.plot(x, y, 'o', label='Original')
    plt.plot(x_new, y_new, label='Interpolated')
    plt.legend()
    plt.xticks(np.arange(12), np.arange(1, 13))
    plt.show()


def Hermite(X, j, x):
    n = len(X)
    l = Lagrange(X, j, x)
    dl = D_Lagrange(X, j, X[j])
    u = (1 - 2 * dl * (x - X[j])) * np.square(l)
    v = (x - X[j]) * np.square(l)
    return u, v


def Lagrange(X, j, x):
    n = len(X)
    l = np.ones(x.shape, dtype=np.double)
    for k in range(0, n):
        if j != k:
            l = l * ((x - X[k]) / (X[j] - X[k]))
    return l


def D_Lagrange(X, j, x):  # Lagranzo daugianario isvestine pagal x
    n = len(X)
    dl = np.zeros(x.shape, dtype=np.double)  # DL israiskos skaitiklis
    for i in range(0, n):  # ciklas per atmetamus narius
        if i == j:
            continue
    lds = np.ones(x.shape, dtype=np.double)
    for k in range(0, n):
        if (k != j) and (k != i):
            lds = lds * (x - X[k])
    dl = dl + lds
    ldv = np.ones(x.shape, dtype=np.double)  # DL israiskos vardiklis
    for k in range(0, n):
        if k != j:
            ldv = ldv * (X[j] - X[k])
    dl = dl / ldv
    return dl


def akima(x, y):
    def fnk(x, xi, xim1, xip1, yi, yim1, yip1):  # 1 akima derivative
        return (2 * x - xi - xip1) / ((xim1 - xi) * (xim1 - xip1)) * yim1 + (2 * x - xim1 - xip1) / (
                (xi - xim1) * (xi - xip1)) * yi + (2 * x - xim1 - xi) / ((xip1 - xim1) * (xip1 - xi)) * yip1

    n = len(x)
    dy = []
    for i in range(n):

        if i == 0:
            xim1 = x[0]
            xi = x[1]
            xip1 = x[2]
            yim1 = [0]
            yi = [1]
            yip1 = [2]
            dy.append(fnk(xim1, xi, xim1, xip1, yi, yim1, yip1))

        elif i == n-1:
            xim1 = x[n - 3]
            xi = x[n - 2]
            xip1 = x[n-1]
            yim1 = y[n - 3]
            yi = y[n - 2]
            yip1 = y[n-1]
            dy.append(fnk(xip1, xi, xim1, xip1, yi, yim1, yip1))

        else:
            xim1 = x[i - 1]
            xi = x[i]
            xip1 = x[i + 1]
            yim1 = y[i - 1]
            yi = y[i]
            yip1 = y[i + 1]
            dy.append(fnk(xi, xi, xim1, xip1, yi, yim1, yip1))

    return np.array(dy)


def interpolate_spline(x, y, dy):
    plt.title('Spain temperatures 2012')
    plt.plot(x, y, 'o', label='original')

    for iii in range(0, len(x) - 1):
        xxx = np.arange(x[iii], x[iii + 1], 0.01)
        fff = 0
        for j in [0, 1]:
            U, V = Hermite([x[iii], x[iii + 1]], j, xxx)
            fff = fff + U * y[iii + j] + V * dy[iii + j]
        if iii == 0:
            plt.plot(xxx, fff, 'b', label='interpolated')
        else:
            plt.plot(xxx, fff, 'b')
    plt.legend()
    plt.xticks(np.arange(12), np.arange(1, 13))
    plt.show()


n = 12  # 12 months
data = pd.read_csv('spain_temperatures_2012.csv')
x = data.index.to_numpy().reshape(-1, 1)  # indexes
y = data.iloc[:, 0].to_numpy().reshape(-1, 1)  # temperatures
del data

interpolate_chebyshev(x, y, n)

interpolate_spline(x, y, akima(x, y))
