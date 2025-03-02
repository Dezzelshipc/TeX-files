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


n = 4 # 0 - 3

Q = 10
alpha = np.array([1, 1, 1, 1]) # n
k = np.array([0.2, 0.2, 0.2]) # n-1, 0<=k_i<=1
m = np.array([1, 1, 1]) # n-1

g = k * alpha[:-1] / alpha[1:]
H = [ g[i%2:i:2] for i in range(n-1) ]

mu = m / alpha[1:]
