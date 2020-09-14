import numpy as np
import matplotlib.pyplot as plt
import texttable as tt

plt.style.use('ggplot')


def f(x):
    return (0.25 * x ** 5) + (0.68 * x ** 4) - (1.65 * x ** 3) - (5.26 * x ** 2) - (1.91 * x) + 1.36


def g(x):
    return (np.e ** -x) * np.cos(x) * np.sin(x ** 2 - 1)


def display_function(intervals, func, title, x_lim, y_lim, x):
    plt.title(title)
    plt.ylim(y_lim)
    plt.xlim(x_lim)
    plt.plot(x, func(x), label='f(x)')

    for interval in intervals:
        plt.axvline(interval[0], linewidth=0.5, color='blue', label='interval')
        plt.axvline(interval[1], linewidth=0.5, color='blue')

    plt.legend()
    plt.show()


def scan_root_intervals(all_roots_interval, func, step=0.1):
    results = []
    x_n = all_roots_interval[0]
    x_n1 = all_roots_interval[0]
    while x_n1 < all_roots_interval[1]:
        x_n1 += step
        if func(x_n1) * func(x_n) < 0:  # detects sign change
            results.append([x_n, x_n1])
        x_n = x_n1
    return results


def chord_method(interval, func, tolerance=1e-8):
    iterations = 0
    x_n = interval[0]
    x_n1 = interval[1]
    x_mid = 0

    while abs(func(x_mid)) > tolerance:
        iterations += 1
        k = abs(func(x_n) / func(x_n1))
        x_mid = (x_n + k * x_n1) / (1 + k)
        if func(x_mid) * func(x_n) > 0:
            x_n = x_mid
        else:
            x_n1 = x_mid

    return iterations, x_mid


def secant_method(starting_points, func, tolerance=1e-8):
    iterations = 0
    x_n = starting_points[0]
    x_n1 = starting_points[1]

    while abs(func(x_n1)) > tolerance:
        iterations += 1
        x_n1 = x_n1 - (x_n1 - x_n) / (func(x_n1) - func(x_n)) * func(x_n1)

    return iterations, x_n1


def scan_method(interval, func, step=0.1, tolerance=1e-8):
    iterations = 1
    x_n = interval[0]
    x_s = interval[0]
    # x_n1 = interval[1]

    # while(x_n1 - x_n) > tolerance:
    while (abs(func(x_s))) > tolerance:
        iterations += 1
        x_s += step
        if func(x_s) * func(x_n) < 0:  # detects sign change
            x_s -= step
            step /= 2
            x_n = x_s
            # x_n1 = x_s + step

    return iterations, x_s


def find_roots(root_intervals, func):
    table = tt.Texttable()
    table.set_cols_dtype(('t', 'a', 'f', 'e', 'i'))
    table.header(('METHOD', 'START', 'RESULT', 'ERROR', 'ITERATIONS'))
    for interval in root_intervals:
        for method in (chord_method, secant_method, scan_method):
            i, result = method(interval, func)
            table.add_row((method.__name__, interval, result, func(result), i))
    print(table.draw())


print("Polynomial")
intervals = scan_root_intervals([-8.64, +5.5869], f)
display_function([[-8.64, +5.5869]], f, 'Starting interval', x_lim=(-10, +10), y_lim=(-2, +2), x=np.arange(-5, +5, 0.1))
display_function(intervals, f, 'Root intervals', x_lim=(-10, +10), y_lim=(-2, +2), x=np.arange(-5, +5, 0.1))
find_roots(intervals, f)

print('Transcendental')
intervals = scan_root_intervals([7, 8], g, 0.003)
display_function(intervals, g, 'Root intervals', x_lim=(7, 8), y_lim=(-0.001, +0.001), x=np.arange(7, 8.01, 0.01))
find_roots(intervals, g)
