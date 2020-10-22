import matplotlib.pyplot as plt
from PyFunkcijos import *
import numpy as np

# gradient of LF function (normalized)
def gradient(x):
    dx0 = (loss([x[0] + 1e-13, x[1]]) - loss(x)) / 1e-13
    dx1 = (loss([x[0], x[1] + 1e-13]) - loss(x)) / 1e-13
    return np.array([dx0, dx1])

def loss(x):
    return (LF(x) ** 2).sum()

def steepest_descent(func, alpha = 0.01):
    x = np.array([-1, 4])

    g = gradient(x)
    for i in range(601):
        prev_loss = loss(x)
        x = x - alpha * g

        if loss(x) > prev_loss:
            x = x + alpha * g  # reverse step
            g = gradient(x)    # recalculate gradient
            x = x - alpha * g  # step in new direction

        if i % 20 == 0:
            print(f'iteration: {i} loss: {loss(x)}')

        if loss(x) < 1e-25:    # good enough
            break

    print()
    print(f'function value: {func(x)}')
    print(f'x = {x}')

# -------- Lygciu sistemos funkcija---------
def LF(x):  # grazina reiksmiu stulpeli
    return np.array([
        [(x[0] ** 2 + x[1] ** 2) / 5 - 2 * cos(x[0] / 2) - 6 * cos(x[1]) - 8],
        [(x[0] / 2) ** 5 + (x[1] / 2) ** 4 - 4]
    ])

steepest_descent(LF)

# ----------------------------------
fig1 = plt.figure(1, figsize=plt.figaspect(0.5))
fig2 = plt.figure(2, figsize=plt.figaspect(0.5))
fig3 = plt.figure(3, figsize=plt.figaspect(0.5))
fig4 = plt.figure(4, figsize=plt.figaspect(0.5))
ax1 = fig1.add_subplot(projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_ylabel('z')
ax2 = fig2.add_subplot(projection='3d')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_ylabel('z')

ax3 = fig3.add_subplot(projection='3d')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_ylabel('z')

ax4 = fig4.add_subplot(projection='3d')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_ylabel('z')

xx = np.linspace(-10, 10, 300)
yy = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(xx, yy)
Z = Pavirsius(X, Y, LF)

surf1 = ax1.plot_surface(X, Y, Z[:, :, 0], color='blue', alpha=0.4)
wire1 = ax1.plot_wireframe(X, Y, Z[:, :, 0], color='black', alpha=1, linewidth=0.3, antialiased=True)

surf1 = ax3.plot_surface(X, Y, Z[:, :, 0], color='blue', alpha=0.4)
wire1 = ax3.plot_wireframe(X, Y, Z[:, :, 0], color='black', alpha=1, linewidth=0.3, antialiased=True)

surf2 = ax1.plot_surface(X, Y, Z[:, :, 1], color='purple', alpha=0.4)
wire2 = ax1.plot_wireframe(X, Y, Z[:, :, 1], color='black', alpha=1, linewidth=0.3, antialiased=True)

surf2 = ax4.plot_surface(X, Y, Z[:, :, 1], color='purple', alpha=0.4)
wire2 = ax4.plot_wireframe(X, Y, Z[:, :, 1], color='black', alpha=1, linewidth=0.3, antialiased=True)

CS11 = ax1.contour(X, Y, Z[:, :, 0], [0], colors='b')
CS12 = ax1.contour(X, Y, Z[:, :, 1], [0], colors='g')
CS1 = ax2.contour(X, Y, Z[:, :, 0], [0], colors='b')
CS2 = ax2.contour(X, Y, Z[:, :, 1], [0], colors='g')

XX = np.linspace(-10, 10, 2)
YY = XX
XX, YY = np.meshgrid(XX, YY)
ZZ = XX * 0
zeroplane = ax2.plot_surface(XX, YY, ZZ, color='gray', alpha=0.4, linewidth=0, antialiased=True)

plt.show()
