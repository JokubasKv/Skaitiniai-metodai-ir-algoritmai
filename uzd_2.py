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

    plt.plot(x, y, 'o', label='Original')
    plt.plot(x_new, y_new, label='Interpolated')
    plt.legend()
    plt.show()


def interpolate_spline(x, y, n):
    pass


n = 12  # 12 months
data = pd.read_csv('spain_temperatures_2012.csv')
x = data.index.to_numpy().reshape(-1, 1)  # indexes
y = data.iloc[:, 0].to_numpy().reshape(-1, 1)  # temperatures

interpolate_chebyshev(x, y, n)

interpolate_spline(x, y, n)
