import numpy as np
from numpy import sin, cos, arccos
import matplotlib.pyplot as plt

plt.style.use('ggplot')


# -2 <= x <= 3 use Chebyshev
def f(x):
    return cos(2 * x) / (sin(2 * x) + 1.5) - x / 5


def chebyshev_polynomial(x, i):
    return cos(i * arccos(x))


def show_plot(x, y):
    plt.plot(x, y)
    plt.show()


def to_chebyshev_interval(x, a, b):
    return (2 * x) / (b - a) - (b + a) / (b - a)


def interpolated(x, a, n):
    xc = to_chebyshev_interval(x, -2, 3)
    A = np.array([chebyshev_polynomial(xc, i) for i in range(len(a))]).T
    return A.dot(a)



n = 10 # crank up
i = np.arange(n)
nodes = np.linspace(-1, 1, n).reshape(-1, 1)
A = chebyshev_polynomial(nodes, i)
data = f(nodes)
coeffs = np.linalg.solve(A, data)
x = np.linspace(-1, 1, 1000).reshape(-1, 1)
A = chebyshev_polynomial(x, i)
plt.plot(x, np.dot(A, coeffs), label='interpolated')
plt.plot(x, f(x), label='original')
plt.legend()
plt.show()









exit()






x = np.arange(-2, 3, 0.1)
# show_plot(x, f(x))

x = np.linspace(-2, 3, 20)
x = np.array([1, 2, 3])
a = x[0]
b = x[-1]

xc = to_chebyshev_interval(x, a, b)
A = np.array([chebyshev_polynomial(xc, i) for i in range(len(xc))]).T
a = np.linalg.solve(A, f(x))
print(A.dot(a))
print(f(x))


xc = to_chebyshev_interval(3, -2, 3)
A = np.array([chebyshev_polynomial(xc, i) for i in range(len(a))]).T
print(A.dot(a))


