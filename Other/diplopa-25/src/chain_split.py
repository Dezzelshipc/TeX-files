import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(function, y0: np.ndarray | float, time_space: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = time_space[1] - time_space[0]
    num = len(time_space)
    x_a = time_space

    y_a = [y0] * (num)

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


# alpha = np.array([20, 16, 12, 8])
# k = np.append([0], [0.3, 0.2, 0.1])
# m = np.append([0], [4, 3, 2])
# a = np.append([0], [0.3, 0.3, 0.3])

# alpha2 = np.array([16, 12, 8])
# k2 = np.array([0.3, 0.2, 0.1])
# m2 = np.array([4, 3, 2])
# a2 = np.array([0.3, 0.3, 0.3])

_q = 4
_r = 2

# alpha = np.array([10] * (_q+1))
# k = np.append([0], [0.5] * _q)
# m = np.append([0], [2] * _q)
# a = np.append([0, 0], [0] * (_q-1))

# alpha2 = np.array([10] * _r)
# k2 = np.array([0.5] * _r)
# m2 = np.array([2] * _r)
# a2 = np.array([0] * _r)

##

alpha = np.linspace(20, 8, _q+1)
k = np.append([0], np.linspace(0.5, 0.3, _q))
m = np.append([0], np.linspace(5, 1, _q))
a = np.append([0, 0.2], [0] * (_q-1))\


alpha_b = 16
alpha2 = np.append([alpha_b], np.linspace(16, 8, _r-1)) 
k2 = np.linspace(0.5, 0.3, _r)
m2 = np.linspace(4, 1, _r)
a2 = np.array([0] * _r)

print("alpha", alpha, alpha2)
print("a", a[1])
print("m", m, m2)
print("k", k, k2)
print()

q = len(m) - 1
r = len(m2)


g = np.append([0], k[1:] * alpha[:-1] / alpha[1:])
H = np.append([1], [ np.prod(g[2-(i%2):i+2:2]) for i in range(1, q+1) ])

mu =np.append([0], m[1:] / alpha[1:])
f = np.append([0], [ sum(mu[2-(i%2):i+2:2]/H[2-(i%2):i+2:2]) for i in range(1, q+1) ])

if q % 2 == 1:
    print("check", f[q], a[1] * m[1] / alpha[0])

cc = np.append(a*m, a2*m2)
alpha = np.append(alpha, alpha2)
k = np.append(k, k2)
m = np.append(m, m2)
a = np.append(a, a2)

cc[2:] = 0
# cc[:] = 0

n = len(alpha)
print(q, r, n)


def get_right_split(func_v, func_1=None):
    func_1 = func_1 or func_v
    def right(_, x):
        return np.array([
            *[Q - alpha[0] * func_1(x[0]) * x[1] + sum(cc * x)],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - x[i+1] * alpha[i] * func_v(x[i])
                for i in range(1, s)
            ],
            *[
                -m[s] * x[s] + k[s] * alpha[s-1] * func_v(x[s-1]) * x[s] - x[s+1] * alpha[s] * func_v(x[s]) - (alpha_b * func_v(x[s]) * x[q+1] if r > 0 else 0)
            ],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < q else 0 ) * alpha[i] * func_v(x[i])
                for i in range(s+1, q+1)
            ],
            *[
                -m[q+1] * x[q+1] + k[q+1] * alpha_b * func_v(x[s]) * x[q+1] - (x[q+2] * alpha[q+1] * func_v(x[q+1]) if r > 1 else 0 )
            ],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < r else 0 ) * alpha[i] * func_v(x[i])
                for i in range(q+2, q+r+1)
            ],
        ])
    return right


def identity(x):
    return x


Q = 200
s = 2

t_s = np.arange(0, 100, 0.01)
N0 = np.array([ 2 ] * (1+q+r))

right_flow = get_right_split(identity)
# right_flow = get_right_flow(np.atan)

Tl, Nl = runge_kutta(right_flow, N0, t_s)

slc_start = 0
slc = -1

start = 0
leg = []

# for i in range(start,n):
#     plt.plot([0, Tl[slc-1]], [NF[i]]*2, "--")
#     leg.append(f"Равн{i}")

for i in range(start,n):
    plt.plot(Tl[slc_start:slc], Nl[slc_start:slc,i])
    leg.append(f"Вид{i if i < q+1 else i-q}")

plt.legend(
    leg, 
    loc='upper right'
)
plt.xlabel('t')
plt.ylabel('N')
plt.title(f"dt={t_s[1]-t_s[0]} {q=} {r=}")

print(Nl[-1])

# plt.savefig("./figs/exp3.pdf")
print(cc[1]/alpha[0])

if (q + 1) % 2 == s % 2:
    if q % 2 == 0:
        N_1 = f[q]
        N_0 = Q / (alpha[0] * N_1) + a[1]* m[1] / alpha[0]
    else:
        N_0 = f[q]
        N_1 = Q / (alpha[0] * N_0 - a[1] * m[1])
    
    print(N_0, N_1)
    plt.plot(t_s[[0, -1]], [N_0]*2, "--")
    plt.plot(t_s[[0, -1]], [N_1]*2, "--")
else:
    pass

plt.show()