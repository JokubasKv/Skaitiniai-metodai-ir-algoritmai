import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def f(x):
    return (0.25 * x ** 5) + (0.68 * x ** 4) - (1.65 * x ** 3) - (5.26 * x ** 2) - (1.91 * x) + 1.36


def show_intervals(intervals, title):
    plt.title(title)
    plt.ylim(-2, +2)
    plt.xlim(-10, +10)
    x = np.arange(-5, +5, 0.1)
    plt.plot(x, f(x), label='f(x)')

    for interval in intervals:
        plt.axvline(interval[0], linewidth=0.5, color='blue', label='interval')
        plt.axvline(interval[1], linewidth=0.5, color='blue')

    plt.legend()
    plt.show()


def scan_root_intervals(all_roots_interval, step=0.1):
    results = []
    previous = all_roots_interval[0]
    x = all_roots_interval[0]
    while x < all_roots_interval[1]:
        x += step
        if f(x) * f(previous) < 0:  # detects sign change
            results.append([previous, x])
        previous = x
    return results


def chord_method(interval, tolerance=1e-8):
    x_n = interval[0]
    x_n1 = interval[1]
    x_mid = 0

    while abs(f(x_mid)) > tolerance:
        k = abs(f(x_n) / f(x_n1))
        x_mid = (x_n + k * x_n1) / (1 + k)
        if f(x_mid) * f(x_n) > 0:
            x_n = x_mid
        else:
            x_n1 = x_mid

    return x_mid


def secant_method(starting_points, tolerance=1e-8):
    x_n = starting_points[0]
    x_n1 = starting_points[1]

    while abs(f(x_n1)) > tolerance:
        x_n1 = x_n1 - (x_n1 - x_n) / (f(x_n1) - f(x_n)) * f(x_n1)

    return x_n1


# show_intervals([[-8.64, +5.5869]], 'Starting interval')

intervals = scan_root_intervals([-8.64, +5.5869])
# show_intervals(intervals, 'Root intervals')
# x = chord_method(intervals[0])
x = secant_method(intervals[0])
print(f'{f(x):.20f}')
