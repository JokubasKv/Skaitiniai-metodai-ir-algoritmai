import matplotlib.pyplot as plt

plt.style.use('ggplot')


def euler():
    t = t0
    v = v0
    h = h0
    m = m1 + m2

    vv = []
    tt = []
    hh = []

    parachute_deployed = False
    while h > 0:
        k = k1 if t < tg else k2
        if parachute_deployed == (t < tg):
            parachute_deployed = True
            print(f'Parachute deployed at {h} meters')

        dv_dt = (m * g - k * v ** 2) / m

        t += dt
        v = v - dt * dv_dt
        h = h + dt * v

        vv.append(v)
        tt.append(t)
        hh.append(h)

    print(f'Hit ground after {t} seconds')
    print(f'Hit ground at {abs(v)} meters per second')

    plt.xlabel('time, (s)')
    plt.ylabel('speed (m/s)')
    plt.plot(tt, vv, )
    plt.show()

    plt.xlabel('time, (s)')
    plt.ylabel('height (m)')
    plt.plot(tt, hh)
    plt.show()


if __name__ == '__main__':
    # parameters
    m1 = 120
    m2 = 10
    h0 = 2500
    tg = 25
    k1 = 0.25
    k2 = 10
    g = 9.8
    dt = 0.001
    t0 = 0
    v0 = 0

    euler()
