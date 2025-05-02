import numpy as np
from scipy.integrate import solve_ivp

def derivatives(t, y, Q, alpha0, a1, m1, s, q, r, m, k, alpha, alpha_s_b, m_prime, k_prime, alpha_prime):
    # Split the state vector into N and N_prime
    N = y[:q+1]
    N_prime = y[q+1: q+1 + r]
    
    # Initialize derivatives
    dNdt = np.zeros(q+1)
    dN_prime_dt = np.zeros(r)
    
    # Compute dN0/dt
    dNdt[0] = Q + N[1] * (-alpha0 * N[0] + a1 * m1)
    
    # Compute dNi/dt for i=1 to s-1 and i=s+1 to q
    # Generate the list of indices i for the first chain
    indices = []
    if s > 1:
        indices += list(range(1, s))
    if s + 1 <= q:
        indices += list(range(s + 1, q + 1))
    
    for i in indices:
        prev_N = N[i-1]
        current_N = N[i]
        next_N = N[i+1] if (i + 1) <= q else 0.0
        term = -m[i] + k[i] * alpha[i-1] * prev_N - alpha[i] * next_N
        dNdt[i] = current_N * term
    
    # Compute dNs/dt if s is within the valid range
    if 0 <= s <= q:
        prev_N = N[s-1] if s > 0 else 0.0  # s=0 would have no previous, but s should be >=1?
        current_N = N[s]
        next_N = N[s+1] if (s + 1) <= q else 0.0
        term = -m[s] + k[s] * alpha[s-1] * prev_N - alpha[s] * next_N - alpha_s_b * N_prime[0]
        dNdt[s] = current_N * term
    
    # Compute dN'_1/dt
    if r >= 1:
        current_N_prime = N_prime[0]
        term = -m_prime[0] + k_prime[0] * alpha_s_b * N[s] 
        if r >= 2:
            term -= alpha_prime[0] * N_prime[1]
        dN_prime_dt[0] = current_N_prime * term
    
    # Compute dN'_k/dt for k=2 to r
    for idx in range(1, r):
        current_N_prime = N_prime[idx]
        term = -m_prime[idx] + k_prime[idx] * alpha_prime[idx-1] * N_prime[idx-1]
        if idx + 1 < r:
            term -= alpha_prime[idx] * N_prime[idx + 1]
        dN_prime_dt[idx] = current_N_prime * term
    
    # Combine the derivatives
    dydt = np.concatenate((dNdt, dN_prime_dt))
    # dydt[dydt == np.inf] = 1e100
    return dydt

# Example parameters (adjust according to your specific problem)
Q = 1000.0
alpha0 = 10
a1 = 0
# a1 = 1
m1 = 2
s = 3
q = 5
r = 3

k_main = 0.5

# Parameters for N_i (i=0 to q)
m = np.zeros(q + 1)  # m[0] unused, m[1] to m[q] used for i=1 to q
m[:] = m1
k = np.ones(q + 1) * k_main
alpha = np.ones(q + 1) * alpha0
alpha_s_b = 2

# Parameters for N'_k (k=1 to r)
m_prime = np.zeros(r)
m_prime[:] = 1
k_prime = np.ones(r) * k_main
alpha_prime = np.ones(r) * 1

# Initial conditions
y0_N = np.ones(q + 1)
y0_N_prime = np.ones(r)
y0 = np.concatenate((y0_N, y0_N_prime))

# Time span and evaluation points
t_span = (0, 1000)
t_eval = np.arange(t_span[0], t_span[1], 0.001)

# method='BDF'
method='RK45'
# Solve the ODE
sol = solve_ivp(
    fun=derivatives,
    t_span=t_span,
    y0=y0,
    t_eval=t_eval,
    args=(Q, alpha0, a1, m1, s, q, r, m, k, alpha, alpha_s_b, m_prime, k_prime, alpha_prime),
    method=method
)

print("alpha", alpha, alpha_prime)
print("a", a1)
print("m", m, m_prime)
print("k", k, k_prime)
print()

# The solution is in sol.y, with time points in sol.t
# sol.y[0 : q+1] are N0 to Nq, sol.y[q+1 : q+1 + r] are N'_1 to N'_r

print(a1 * m1 / alpha0)
print(sol.y[:q+1,-1])
print(sol.y[q+1:,-1])

from matplotlib import pyplot as plt

for s in sol.y:
    plt.plot(t_eval, s)

plt.legend(range(q+r))

plt.title(f"{method} {t_eval[1] - t_eval[0]}")
plt.show()