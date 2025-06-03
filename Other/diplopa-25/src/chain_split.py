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


_q = 4
_r = 2

##

# alpha = np.linspace(20, 10, _q+1)
# k = np.append([0], np.linspace(0.5, 0.2, _q))
# m = np.append([0], np.linspace(5, 2, _q))
# a = np.append([0, 0.2], [0] * (_q-1))\

# alpha_b = 16
# alpha2 = np.append([alpha_b], np.linspace(16, 8, _r)) 
# k2 = np.append([0], np.linspace(0.5, 0.3, _r))
# m2 = np.append([0], np.linspace(4, 1, _r))
# a2 = np.append([0, 0.0], np.array([0] * (_r-1)))


# alpha = np.linspace(80, 20, _q+1)
# k = np.append([0], np.linspace(0.8, 0.5, _q))
# m = np.append([0], np.linspace(20, 10, _q))
# a = np.append([0, 0.2], [0] * (_q-1))\

# alpha_b = 20
# alpha2 = np.append([alpha_b], np.linspace(16, 8, _r)) 
# k2 = np.append([0], np.linspace(0.5, 0.3, _r))
# m2 = np.append([0], np.linspace(4, 2, _r))
# a2 = np.append([0, 0.0], np.array([0] * (_r-1)))


alpha = np.linspace(20, 10, _q+1)
k = np.append([0], np.linspace(0.5, 0.2, _q))
m = np.append([0], np.linspace(5, 2, _q))
a = np.append([0, 0.2], [0] * (_q-1))\

alpha_b = 16
alpha2 = np.append([alpha_b], np.linspace(16, 8, _r)) 
k2 = np.append([0], np.linspace(0.5, 0.3, _r))
m2 = np.append([0], np.linspace(4, 1, _r))
a2 = np.append([0, 0.0], np.array([0] * (_r-1)))

print("alpha", alpha, alpha2)
print("a", a)
print("m", m, m2)
print("k", k, k2)
print()

# exit()

g = np.append([0], k[1:] * alpha[:-1] / alpha[1:])
H = np.append([1], [ np.prod(g[2-(i%2):i+2:2]) for i in range(1, _q+1) ])

mu =np.append([0], m[1:] / alpha[1:])
f = np.append([0], [ sum(mu[2-(i%2):i+2:2]/H[2-(i%2):i+2:2]) for i in range(1, _q+1) ])

g2 = np.append([0], k2[1:] * alpha2[:-1] / alpha2[1:])
H2 = np.append([1], [ np.prod(g2[2-(i%2):i+2:2]) for i in range(1, _q+1) ])

mu2 =np.append([0], m2[1:] / alpha2[1:])
f2 = np.append([0], [ sum(mu2[2-(i%2):i+2:2]/H2[2-(i%2):i+2:2]) for i in range(1, _r+1) ])

print(f"{f=}")
print(f"{f2=}")


alpha = np.append(alpha, alpha2[1:])
k = np.append(k, k2[1:])
m = np.append(m, m2[1:])
a = np.append(a, a2[1:])

cc = a*m

print(cc)

cc[2:] = 0
# cc[:] = 0

n = len(alpha)
print(_q, _r, n)


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
                -m[s] * x[s] + k[s] * alpha[s-1] * func_v(x[s-1]) * x[s] - x[s+1] * alpha[s] * func_v(x[s]) - (alpha_b * func_v(x[s]) * x[_q+1] if _r > 0 else 0)
            ],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < _q else 0 ) * alpha[i] * func_v(x[i])
                for i in range(s+1, _q+1)
            ],
            *[
                -m[_q+1] * x[_q+1] + k[_q+1] * alpha_b * func_v(x[s]) * x[_q+1] - (x[_q+2] * alpha[_q+1] * func_v(x[_q+1]) if _r > 1 else 0 )
            ],
            *[
                -m[i] * x[i] + k[i] * alpha[i-1] * func_v(x[i-1]) * x[i] - (x[i+1] if i < _q+_r else 0 ) * alpha[i] * func_v(x[i])
                for i in range(_q+2, _q+_r+1)
            ],
        ])
    return right


def identity(x):
    return x


### CHANGE PARAMETERS
Q = 100

s = 2

q = 2
r = 0
###

t_s = np.arange(0, 100, 0.001)
N0 = np.array([ 0.5 ] * (1+_q+_r))

# right_flow = get_right_split(identity)
right_flow = get_right_split(np.atan)

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
    leg.append(f"Вид{i if i < _q+1 else f"'{i-_q}"}")

print(Nl[-1])


N_0 = N_1 = None

def N0_f(_N_1):
    return Q / (alpha[0] * _N_1) + a[1]* m[1] / alpha[0]

def N1_f(_N_0):
    return Q / (alpha[0] * _N_0 - a[1] * m[1])

if (q + 1) % 2 == s % 2:
    print(1)
    X = f[q]
    if q % 2 == 0:
        N_1 = X
        N_0 = N0_f(X)
    else:
        N_0 = X
        N_1 = N1_f(X)
    

elif r % 2 == 0 and q % 2 == s % 2:
    print(2)
    X = f[q] + alpha_b/alpha[s] * f2[r]/H[s]
    if s % 2 == 0:
        N_1 = X
        N_0 = N0_f(X)
    else:
        N_0 = X
        N_1 = N1_f(X)

elif r % 2 == 1:
    print(3)
    X = f2[r] / H[s-1] + f[s-1]
    if s % 2 == 0:
        N_0 = X
        N_1 = N1_f(X)
    else:
        N_0 = N0_f(X)
        N_1 = X
else:
    print("__ !! NOT HANDLED !! __")


print(N_0, N_1)

### PLOT STABLE STATES
if N_0 and N_1:
    # plt.plot(t_s[[0, -1]], [N_0]*2, "--")
    # plt.plot(t_s[[0, -1]], [N_1]*2, "--")

    leg.extend([
        "Равн0",
        "Равн1"
    ])

plt.legend(
    leg, 
    loc='upper right'
)
plt.xlabel('t')
plt.ylabel('N')
plt.title(f"{q=} {r=} {s=}")
print(f"dt={t_s[1] - t_s[0]}")

### PLOT STABLE STATES


### CALCULATE Q THRESHOLD

f = np.append(f, [f[-2], f[-1]])
f2 = np.append(f2, [f2[-2], f2[-1]])
# f = np.append(f, [0]*2)
# f2 = np.append(f2, [0]*2)

def calc_Q(ss, qq, rr):
    QQ1, QQ2, QR1, QR2 = (0,)*4

    u = ss // 2
    p = qq // 2
    l = rr // 2

    alpha0 = alpha[0]
    a1 = a[1]
    m1 = m[1]
    alphasb_alphas = alpha_b / alpha[ss]

    if ss % 2 == 0:

        if qq % 2 == 0 and rr % 2 == 0:
            QQ1 = (alpha0 * f[2*p-1] - a1 * m1) * (f[2*p] + alphasb_alphas * f2[2*l] / H[2*u])
            QQ2 = (alpha0 * f[2*p+1] - a1 * m1) * (f[2*p] + alphasb_alphas * f2[2*l] / H[2*u])

            QR1 = (alpha0 * f[2*u-1] - a1 * m1 + alpha0 * f2[2*l-1] / H[2*u-1]) * (f[2*p] + alphasb_alphas * f2[2*l] / H[2*u])
            QR2 = (alpha0 * f[2*u-1] - a1 * m1 + alpha0 * f2[2*l+1] / H[2*u-1]) * (f[2*p] + alphasb_alphas * f2[2*l] / H[2*u])
        elif qq % 2 == 1 and rr % 2 == 0:
            QQ1 = (alpha0 * f[2*p+1] - a1 * m1) * (f[2*p] + alphasb_alphas * f2[2*l] / H[2*u])
            QQ2 = (alpha0 * f[2*p+1] - a1 * m1) * (f[2*p+2] + alphasb_alphas * f2[2*l] / H[2*u])
            
            QR1 = f[2*p+1] - f[2*u-1] - f2[2*l+1] / H[2*u-1]
            QR2 = f[2*p+1] - f[2*u-1] - f2[2*l-1] / H[2*u-1]
            QR1 = (QR1, True)
            QR2 = (QR2, True)
        elif qq % 2 == 0 and rr % 2 == 1: 
            QQ1 = f[2*p-1] - f[2*u-1] - f2[2*l+1] / H[2*u-1]
            QQ2 = f[2*p+1] - f[2*u-1] - f2[2*l+1] / H[2*u-1]
            QQ1 = (QQ1, True)
            QQ2 = (QQ2, True)

            QR1 = (alpha0 * f[2*u-1] - a1 * m1 + alpha0 * f2[2*l+1] / H[2*u-1]) * (f[2*p] + alphasb_alphas * f2[2*l] / H[2*u])
            QR2 = (alpha0 * f[2*u-1] - a1 * m1 + alpha0 * f2[2*l+1] / H[2*u-1]) * (f[2*p] + alphasb_alphas * f2[2*l+2] / H[2*u])
        elif qq % 2 == 1 and rr % 2 == 1:
            QQ1, QQ2, QR1, QR2 = (-1,)*4
        
    else:
        if qq % 2 == 0 and rr % 2 == 0:
            QQ1 = f[2*p] * (alpha0 * f[2*p-1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1])
            QQ2 = f[2*p] * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1])
            
            QR1 = f[2*p] - f[2*u] - f2[2*l+1] / H[2*u]
            QR2 = f[2*p] - f[2*u] - f2[2*l-1] / H[2*u]
            QR1 = (QR1, True)
            QR2 = (QR2, True)
        elif qq % 2 == 1 and rr % 2 == 0:
            QQ1 = f[2*p] * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1])
            QQ2 = f[2*p+2] * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1])

            QR1 = (f[2*u] + f2[2*l-1] / H[2*u]) * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1]) 
            QR2 = (f[2*u] + f2[2*l+1] / H[2*u]) * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1])
        elif qq % 2 == 1 and rr % 2 == 1: 
            QQ1 = f[2*p] - f[2*u] - f2[2*l+1] / H[2*u]
            QQ2 = f[2*p+2] - f[2*u] - f2[2*l+1] / H[2*u]
            QQ1 = (QQ1, True)
            QQ2 = (QQ2, True)

            QR1 = (f[2*u] + f2[2*l+1] / H[2*u]) * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l] / H[2*u+1]) 
            QR2 = (f[2*u] + f2[2*l+1] / H[2*u]) * (alpha0 * f[2*p+1] - a1 * m1 + alpha0 * alphasb_alphas * f2[2*l+2] / H[2*u+1])
        elif qq % 2 == 0 and rr % 2 == 1:
            QQ1, QQ2, QR1, QR2 = (-1,)*4


    return QQ1, QQ2, QR1, QR2

def calc_Q_table(s_, q_, r_):
    print("---")
    for qi in range(1,q_+1):
        for ri in range(r_+1):
            QQ1, QQ2, QR1, QR2 = calc_Q(s_, qi, ri)
            print(f"q={qi}: {QQ1} < Q < {QQ2}")
            print(f"r={ri}: {QR1} < Q < {QR2}")
            print("---")

def calc_Q_table2(s_, q_, r_):
    def trunc(x, pow: int):
        from numpy import trunc as trnc
        p10 = 10**int(pow)
        return trnc(x * p10) / p10

    print("-----")
    print(r"""\hline
\backslashbox{\(q\)}{\(r\)} & """ + f"{" & ".join(fr"\({i}\)" for i in range(r_+1))}" + r" \\ \hline")
    for qi in range(1,q_+1):
        print(fr"\({qi}\)")
        for ri in range(r_+1):
            QQ1, QQ2, QR1, QR2 = calc_Q(s_, qi, ri)
            out_str = None
            if QQ1 == -1:
                out_str = "& -- "
            else:
                qstr, rstr = "", ""
                if isinstance(QQ1, tuple):
                    q1 = trunc(QQ1[0], 2)
                    q2 = trunc(QQ2[0], 2)
                    qstr = fr"q?({q1}, {q2})"
                else:
                    q1 = trunc(QQ1, 2)
                    q2 = trunc(QQ2, 2)
                    if QQ1 == QQ2:
                        q2 = r"+\infty"
                    qstr = fr"q({q1}, {q2})"

                if isinstance(QR1, tuple):
                    r1 = trunc(QR1[0], 2)
                    r2 = trunc(QR2[0], 2)
                    rstr = fr"r?({r1}, {r2})"
                else:
                    r1 = trunc(QR1, 2)
                    r2 = trunc(QR2, 2)
                    if QR1 == QR2:
                        r2 = r"+\infty"
                    rstr = fr"r({r1}, {r2})"

                out_str = fr"& \(\begin{{matrix}} {qstr} \\ {rstr} \end{{matrix}}\) "

            if ri == r_:
                out_str += r"\\ \hline"

            print(out_str)
    print("-----")


# calc_Q_table2(s, _q, _r)

print("Current threshold")
QQ1, QQ2, QR1, QR2 = calc_Q(s, q, r)
print(f"{q=}: {QQ1} < Q < {QQ2}")
print(f"{r=}: {QR1} < Q < {QR2}")

###

print(f"{s=}, {_q=}, {_r=}, {Q=}")

add_ = "exp2_"

from pathlib import Path
# plt.savefig(Path(__file__).parent / f"figs/{add_}s{s}_{alpha_b}_Q{Q}.pdf")


plt.show()