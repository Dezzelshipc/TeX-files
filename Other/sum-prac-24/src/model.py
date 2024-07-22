import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sy


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h)
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * (num)

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


ksi1, ksi2, ksi3 = 10, 8, 6
a12, a13, a23 = 6, 2, 0.5
k12, k13, k23 = 4, 1, 0.5


def static_point(num: int, num2=None):
    match (num, num2):
        case (0, None):
            return [0, 0, 0]
        case (1, None) | (1, 2) | (2, 1):
            return [0, ksi3 / (k23 * a23), ksi2 / a23]
        case (2, None) | (0, 2) | (2, 0):
            return [ksi3 / (k13 * a13), 0, ksi1 / a13]
        case (3, None) | (0, 1) | (1, 0):
            return [-ksi2 / (k12 * a12), ksi1 / a12, 0]
        case (4, None):
            ld = k13 - k12 * k23
            if ld != 0:
                return np.array([
                    (ksi3 * a12 - ksi1 * a23 * k23 + ksi2 * k23 * a13) / (a12 * a13 * ld),
                    (ksi1 * a13 * k13 - ksi2 * a13 * k13 - ksi3 * a12 * k12) / (a12 * a23 * ld),
                    (ksi3 * a12 * k12 + ksi2 * a13 * k13 - ksi1 * k23 * k12 * k23) / (a13 * a23 * ld),
                ])
            else:
                x3 = 1
                return np.array([
                    (a23 * x3 - ksi2) / (a12 * k12),
                    (-a13 * x3 + ksi1) / a12,
                    x3,
                ])


def right1(t, x):
    return np.array([
        (ksi1 - a12 * x[1] - a13 * x[2]) * x[0],
        (ksi2 + k12 * a12 * x[0] - a23 * x[2]) * x[1],
        (-ksi3 + k13 * a13 * x[0] + k23 * a23 * x[1]) * x[2]
    ])


def right(t, x):
    return np.array([
        ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] - 3 * x[2] * x[0]),
        ((1 * x[0] - 5) * x[1] - 1 * x[2] * x[1]),
        ((1 * x[0] - 3) * x[2] + (1 * x[1] - 4) * x[2])
    ])


def plot3(tl, xl):
    plt.figure(0)
    plt.plot(tl, xl[0])
    plt.plot(tl, xl[1])
    plt.plot(tl, xl[2])

    plt.legend(["x1", "x2", "x3"])


def plotp(tl, xl, n1, n2):
    plt.figure(f"{n1}{n2}")
    plt.plot(xl[n1], xl[n2], 'o-', markevery=[0])
    plt.xlabel(f"x{n1 + 1}")
    plt.ylabel(f"x{n2 + 1}")
    st_point = static_point(n1, n2)
    plt.plot(st_point[n1], st_point[n2], 'o')


def plotp3(tl, xl):
    ax = plt.figure("123").add_subplot(projection='3d')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

    ax.plot(*static_point(0), 'o')
    ax.plot(*static_point(1), 'o')
    ax.plot(*static_point(2), 'o')
    ax.plot(*static_point(3), 'o')
    # ax.plot(*static_point(4), 'o')

    ax.legend(["x(0)", "x(1)", "x(2)", "x(3)", "x(4)"])

    ax.plot(xl[0], xl[1], xl[2], 'o-', markevery=[0])


def plotvec(n1, n2, lin1, lin2):
    plt.figure(f"vec{n1}{n2}")
    plt.xlabel(f"x{n1 + 1}")
    plt.ylabel(f"x{n2 + 1}")
    # plt.plot(*static_point(n1, n2), 'o')

    stat = filter(lambda x: x[0]>= 0 and x[1]>=0 and x[2]>=0, statics)
    
    grid = np.meshgrid(lin1, lin2)

    x1e = 0
    x2e = 0
    x3e = 0
    if 0 not in (n1, n2):
        vec_grid = right(0, [x1e, grid[0], grid[1]])
        stat = filter(lambda x: x[0] == x1e, stat)

    elif 1 not in (n1, n2):
        vec_grid = right(0, [grid[0], x2e, grid[1]])
        stat = filter(lambda x: x[1] == x2e, stat)


    elif 2 not in (n1, n2):
        vec_grid = right(0, [grid[0], grid[1], x3e])
        stat = filter(lambda x: x[2] == x3e, stat)

    plt.streamplot(grid[0], grid[1], vec_grid[n1], vec_grid[n2], color='black', broken_streamlines=False, density=0.5)
    
    plt.streamplot(grid[0], grid[1], vec_grid[n1], vec_grid[n2], color='black', broken_streamlines=False, start_points=[[10,0.01]])

    stat = list(stat)
    for st_point in stat:
        print(st_point)
        plt.plot(st_point[n1], st_point[n2], 'o')


def plotvec3(linx, liny, linz):
    ax = plt.figure("vector field").add_subplot(projection='3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

    grid = np.meshgrid(linx, liny, linz)
    vec_grid = right(0, grid)

    ax.quiver(*grid, *vec_grid, length=0.01, color='black')

    # ax.plot(*static_point(0), 'o')
    # ax.plot(*static_point(1), 'o')
    ax.plot(*static_point(2), 'o')
    # ax.plot(*static_point(3), 'o')

    # ax.set_xlim(-0.1, 12)
    # ax.set_ylim(-0.1, 20)
    # ax.set_zlim(-3, 3)


    
x1, x2, x3 = sy.symbols('x1 x2 x3')
x = (x1,x2,x3)

m = right(0, x)

statics = sy.solve(m,x1, x2, x3)
print(statics)


a, b = 0, 10
n = 10000
h = (b - a) / n

x0 = np.array([10, 10, 1])
# x0 = static_point()
# tl, xl = runge_kutta(right, x0, a, b, h)
# xl = xl.T

# print(x0)

# plot3(tl, xl)

# plotp(tl, xl, 0, 1)
# plotp(tl, xl, 0, 2)
# plotp(tl, xl, 1, 2)

# plotp3(tl, xl)

l1 = np.linspace(0, 20, 100)

plotvec(0, 1, l1, l1)
plotvec(0, 2, l1, l1)
plotvec(1, 2, l1, l1)
# plotvec3(np.linspace(0.01, 40, 21), np.linspace(0.01, 40, 21), 5)

# print(x0:=static_point(4))
# a2 = k12 * a12**2 * x0[0] * x0[1] + k13 * a13**2 * x0[0]*x0[2] + k23 * a23 ** 2 * x0[1] * x0[2]
# print(f"{a2=}")
# a3 = x0[0] * x0[1] * x0[2] * a12 * a13* a23 * (k13- k12 * k23)
# print(f"{a3=}")


plt.show()
