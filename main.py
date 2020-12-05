import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
plt.style.use('ggplot')


def show_results(vv, tt, hh, v, t, h, title):
    def plot(x, y, y_label):
        plt.title(title)
        plt.xlabel('time, (s)')
        plt.ylabel(y_label)
        plt.plot(x, y)
        plt.show()

    print(title)
    print(f'Parachute deployed at {h} meters')
    print(f'Hit ground after {t} seconds')
    print(f'Hit ground at {abs(v)} meters per second\n')

    plot(tt, vv, 'speed (m/s)')
    plot(tt, hh, 'height (m)')


def euler():
    t = t0
    v = v0
    h = h0
    m = m1 + m2

    vv = []
    tt = []
    hh = []

    ddd = []

    parachute_deployed = False
    hp = -1
    while h > 0:
        k = k1 if t < tg else k2
        if parachute_deployed == (t < tg):
            parachute_deployed = True
            hp = h

        dv_dt = (m * g + k * v * np.abs(v)) / m
        ddd.append(dv_dt)
        t += dt
        v = v - dt * dv_dt
        h = h + dt * v

        vv.append(v)
        tt.append(t)
        hh.append(h)

    show_results(vv, tt, hh, v, t, hp, 'Euler')

def rk_4():
    t = t0
    v = v0
    h = h0
    m = m1 + m2

    vv = []
    tt = []
    hh = []

    parachute_deployed = False
    hp = -1
    while h > 0:
        k = k1 if t < tg else k2
        if parachute_deployed == (t < tg):
            parachute_deployed = True
            hp = h

        t += dt

        dv_dt1 = (m * g + k * v * np.abs(v)) / m
        v1 = v - dt / 2 * dv_dt1
        dv_dt2 = (m * g + k * v1 * np.abs(v1)) / m
        v2 = v - dt / 2 * dv_dt2
        dv_dt3 = (m * g + k * v2 * np.abs(v2)) / m
        v3 = v - dt * dv_dt3

        dv_dt4 = (m * g + k * v3 * np.abs(v3)) / m
        v = v - dt / 6 * (dv_dt1 + 2 * dv_dt2 + 2 * dv_dt3 + dv_dt4)

        h = h + dt * v

        vv.append(v)
        tt.append(t)
        hh.append(h)

    show_results(vv, tt, hh, v, t, hp, 'RK4')


def model(v, t):
    m = m1+m2
    k = k1 if t < tg else k2
    dv_dt = (m * g - k * v * np.abs(v)) / m
    return dv_dt


if __name__ == '__main__':
    # parameters
    m1 = 120
    m2 = 10
    h0 = 2500
    tg = 25
    k1 = 0.25
    k2 = 10
    g = 9.8
    t0 = 0
    v0 = 0


    dt = 0.01
    euler()
    rk_4()

    # x = np.arange(0,120, 0.0001)
    # plt.plot(x, -odeint(model, 0, x))
    # plt.show()
