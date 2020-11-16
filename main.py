import numpy as np
from numpy import sin, cos, arccos
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# -2 <= x <= 3 use Chebyshev
def f(x):
    return cos(2 * x) / (sin(2 * x) + 1.5) - x / 5


def chebyshev_polynomial(x, i):
    return cos(i * arccos(x))


def to_chebyshev_interval(x, a, b):
    return (2 * x) / (b - a) - (b + a) / (b - a)


n = 15
i = np.arange(n)

x = np.linspace(-2, 3, n).reshape(-1, 1)
x_chebyshev = to_chebyshev_interval(x, -2, 3)
A = chebyshev_polynomial(x_chebyshev, i)
coefficients = np.linalg.solve(A, f(x))


x = np.linspace(-2, 3, 1000).reshape(-1, 1)
xc = to_chebyshev_interval(x, -2, 3)
A = chebyshev_polynomial(xc, i)
plt.plot(x, A.dot(coefficients), label='interpolated')
plt.plot(x, f(x), label='original')
plt.legend()
plt.show()
