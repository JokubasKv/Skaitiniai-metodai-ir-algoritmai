import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def sample(arr, n):
    indexes = np.arange(0, len(arr), len(arr) / n).astype(np.int)
    return np.take(arr, indexes)


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


def Hermite(X, j, x):
    n = len(X)
    l = Lagrange(X, j, x)
    dl = D_Lagrange(X, j, X[j])
    u = (1 - 2 * dl * (x - X[j])) * np.square(l)
    v = (x - X[j]) * np.square(l)
    return u, v


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

        elif i == n - 1:
            xim1 = x[n - 3]
            xi = x[n - 2]
            xip1 = x[n - 1]
            yim1 = y[n - 3]
            yi = y[n - 2]
            yip1 = y[n - 1]
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


def interpolate_spline(t, x, y, dx, dy):
    for iii in range(0, len(t) - 1):
        ttt = np.arange(t[iii], t[iii + 1], 0.01)
        fffx = 0
        fffy = 0
        for j in [0, 1]:
            U, V = Hermite([t[iii], t[iii + 1]], j, ttt)
            fffx = fffx + U * x[iii + j] + V * dx[iii + j]
            fffy = fffy + U * y[iii + j] + V * dy[iii + j]
        if iii == 0:
            plt.plot(fffx, fffy, 'b', label='interpolated')
        else:
            plt.plot(fffx, fffy, 'b')

    plt.legend()
    plt.show()


data = pd.read_csv('spain_border_cropped.csv')
x = data.iloc[:, 1].to_numpy().reshape(-1, 1)
y = data.iloc[:, 0].to_numpy().reshape(-1, 1)
x = np.append(x, x[0])
y = np.append(y, y[0])
del data

n = int(input("Specify number of interpolation points: "))
x = sample(x, n)
y = sample(y, n)

plt.plot(x, y, 'ro', label=f'original (n={n})')
plt.title('Spain contour')

# add first point to the end to have closed curve
x = np.append(x, x[0])
y = np.append(y, y[0])
n += 1

t = np.arange(0, n, 1).reshape(-1, 1)
interpolate_spline(t, x, y, akima(t, x), akima(t, y))
