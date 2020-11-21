import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def g(x, n):
    x = x.flatten()
    return np.array([x ** i for i in range(n)]).T.astype(np.float)


def f(x, coefficients):
    coefficients = coefficients.flatten()
    result = 0
    for i in range(len(coefficients)):
        result += coefficients[i] * x ** i
        print(f'{coefficients[i]} * x ^ {i} +')
    return result


n = int(input('Enter desired degree of polynomial: '))
n += 1
data = pd.read_csv('spain_temperatures_2012.csv')
x = data.index.to_numpy().reshape(-1, 1)  # indexes
y = data.iloc[:, 0].to_numpy().reshape(-1, 1)  # temperatures
del data

G = g(x, n)
coefficients = np.linalg.solve(G.T.dot(G), G.T.dot(y))

xx = np.arange(0, 11.1, 0.1).reshape(-1,1)
plt.plot(x, y, 'o', label=f'original\nn={len(x)}')
plt.plot(xx, f(xx, coefficients), label=f'approximation\nn={n-1}')
plt.xticks(np.arange(12), np.arange(1, 13))
plt.title('Spain temperatures 2012')
plt.legend()
plt.show()
