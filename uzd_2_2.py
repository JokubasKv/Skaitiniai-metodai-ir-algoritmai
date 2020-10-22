import numpy as np


def LF(x):
    return np.array([
        2 * x[1] + x[2] + 2 * x[3] - 14,
        3 * x[3] ** 3 + 3 * x[1] * x[3] + 18,
        -2 * x[0] ** 2 + 5 * x[1] ** 3 - 3 * x[2] ** 2 - 485,
        5 * x[0] - 6 * x[1] + 3 * x[2] - 3 * x[3] - 11
    ])


def loss(x):
    return (LF(x) ** 2).sum()

def gradient(x):
    dx0 = (loss([x[0] + 1e-14, x[1], x[2], x[3]]) - loss(x)) / 1e-14
    dx1 = (loss([x[0], x[1] + 1e-14, x[2], x[3]]) - loss(x)) / 1e-14
    dx2 = (loss([x[0], x[1], x[2] + 1e-14, x[3]]) - loss(x)) / 1e-14
    dx3 = (loss([x[0], x[1], x[2], x[3] + 1e-14]) - loss(x)) / 1e-14
    g = np.array([dx0, dx1, dx2, dx3])
    return g

# 4 5 6 -1
def steepest_descent(alpha=1.1):
    # x = np.array([3.9, 4.9, 5.9, -0.9])
    # x = np.array([4.5, 5.5, 6.5, -0.5])
    x = np.array([1, 1, 1, 1])
    # print(gradient(x) * alpha)
    # exit()

    right_steps = 0
    g = gradient(x)
    for i in range(1_000_000):
        l = loss(x)
        if l < 1e-17:  # very good in this case
            break

        x = x - alpha * g

        if l < loss(x):
            x = x + alpha * g
            g = gradient(x)
            alpha *= 0.4
        else:
            right_steps += 1
            if right_steps > 10:
                alpha *= 10000
                right_steps = 0

        print(f'loss: {loss(x)} after iteration: {i}')
    print()
    print(f'function value: {LF(x)}')
    print(f'x = {x}')

steepest_descent()