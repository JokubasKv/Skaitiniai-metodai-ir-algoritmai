import numpy as np
from numpy import sin, cos, arccos, pi
import matplotlib.pyplot as plt
import sympy as sym


plt.style.use('ggplot')


# -2 <= x <= 3 use Chebyshev
def f(x):
    return cos(2 * x) / (sin(2 * x) + 1.5) - x / 5


def chebyshev_polynomial(x, i):
    return cos(i * arccos(x))

def symbolic_chebyshev_polynomial(x, i):
    return sym.cos(i * sym.acos(x))

def to_chebyshev_interval(x, a, b):
    return (2 * x) / (b - a) - (b + a) / (b - a)


def chebyshev_node(i, a, b, n):
    """
    :param i: iteration (Can be numpy array)
    :param a: interval start
    :param b: interval end
    :param n: number of nodes
    """
    return ((b - a) / 2) * cos(pi * (2 * i + 1) / (2 * n)) + ((b + a) / 2)


def show_symbolic_chebyshev(i, coeffs):
    x = sym.Symbol('x')
    coeffs = coeffs.flatten()
    xc = to_chebyshev_interval(x, -2, 3)
    for i in range(len(coeffs)):
        A = symbolic_chebyshev_polynomial(xc, i)
        if i == 0:
            print(str(coeffs[i]) + ' +')
        # elif i == 1:
        #     print(str(coeffs[i]) + ' * (' + str(A) + ') +')
        else:
            print(str(coeffs[i]) + ' * ' + str(A) + ' +')


n = 15
i = np.arange(n)

x = np.linspace(-2, 3, n).reshape(-1, 1); plot_name = 'Equidistant nodes'  # use equidistant nodes
# x = chebyshev_node(i, -2, 3, n).reshape(-1, 1); plot_name = 'Chebyshev nodes'  # use chebyshev nodes

x_chebyshev = to_chebyshev_interval(x, -2, 3)
A = chebyshev_polynomial(x_chebyshev, i)
coefficients = np.linalg.solve(A, f(x))

x = np.linspace(-2, 3, 1000).reshape(-1, 1)

# show_symbolic_chebyshev(i, coefficients)
xc = to_chebyshev_interval(x, -2, 3)

A = chebyshev_polynomial(xc, i)
interpolated = A.dot(coefficients)


plt.plot(x, interpolated, label='Interpolated')
plt.plot(x, f(x), label='Original')
plt.plot(x, f(x)-interpolated, label='Error')
plt.legend()
plt.title(plot_name)
plt.show()
