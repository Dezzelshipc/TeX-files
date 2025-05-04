import sympy as sp
from sympy import IndexedBase, symbols, Symbol, Poly
from functools import lru_cache
from tbcontrol.symbolic import routh

def compute_routh_hurwitz(q_val):
    # Define all necessary symbols and indexed bases
    alpha = IndexedBase('alpha')
    k = IndexedBase('k')
    a = IndexedBase('a')
    m = IndexedBase('m')
    Q = Symbol('Q')
    
    s = Symbol('s')  # Polynomial variable

    # Compute g_i and mu_i
    g = {}
    mu = {}
    for i in range(1, q_val + 1):
        g[i] = k[i] * alpha[i-1] / alpha[i]
        mu[i] = m[i] / alpha[i]

    # Compute H and f dictionaries
    H = {}
    f = {}
    max_s = (q_val) // 2
    for s_val in range(1, max_s + 1):
        # Compute H_{2s-1} and H_{2s}
        H_odd = 1
        for j in range(1, 2*s_val, 2):
            H_odd *= g[j]
        H[2*s_val - 1] = H_odd

        H_even = 1
        for j in range(2, 2*s_val+1, 2):
            H_even *= g[j]
        H[2*s_val] = H_even

        # Compute f_{2s-1} and f_{2s}
        f_odd = 0
        for j in range(1, s_val+1):
            term = mu[2*j - 1] / H[2*j - 1]
            f_odd += term
        f[2*s_val - 1] = f_odd

        f_even = 0
        for j in range(1, s_val+1):
            term = mu[2*j] / H[2*j]
            f_even += term
        f[2*s_val] = f_even

    # Determine if q is even or odd and compute N0 and N1
    N = {}
    if q_val % 2 == 0:
        s = q_val // 2
        N1 = f[2*s]
        N0 = Q / (alpha[0] * N1)
    else:
        s = (q_val - 1) // 2
        N0 = f[2*s + 1]
        N1 = Q / (alpha[0] * N0)
    N[0] = N0
    N[1] = N1

    # Compute remaining N_i
    for i in range(2, q_val + 1):
        if i % 2 == 0:
            j = i // 2
            H_key = 2*j - 1
            f_key = 2*j - 1
            N_i = H[H_key] * (N[0] - f[f_key])
        else:
            j = (i - 1) // 2
            H_key = 2*j
            f_key = 2*j
            N_i = H[H_key] * (N[1] - f[f_key])
        N[i] = N_i

    # Compute b_i, d_i, c_i
    b = {0: alpha[0] * N[1]}
    d = {0: alpha[0] * N[0]}
    c = {}
    for i in range(1, q_val + 1):
        b[i] = k[i] * alpha[i-1] * N[i]
        d[i] = alpha[i] * N[i]
        c[i] = a[i] * m[i]

    # Recursive computation of e_i(q) with memoization
    memo_e = {}
    # Base cases
    memo_e[(0, 0)] = b[0]
    memo_e[(1, 1)] = b[0]
    if q_val >= 1:
        memo_e[(0, 1)] = b[1] * (d[0] - c[1])

    def e(i, q_current):
        key = (i, q_current)
        if key in memo_e:
            return memo_e[key]
        if i == q_current + 1:
            return 1
        if i >= q_current + 2:
            return 0
        if i == 0:
            product_b = 1
            for j in range(1, q_current + 1):
                product_b *= b[j]
            term1 = b[q_current] * d[q_current - 1] * e(0, q_current - 2)
            term2 = (-1)**q_current * c[q_current] * product_b
            result = term1 - term2
        elif 1 <= i <= q_current:
            term1 = e(i-1, q_current - 1)
            term2 = b[q_current] * d[q_current - 1] * e(i, q_current - 2)
            result = term1 + term2
        else:
            result = 0
        memo_e[key] = result
        return result

    # Generate polynomial coefficients
    s = Symbol('s')
    coefficients = [e(i, q_val) for i in range(q_val + 2)]
    print(coefficients)

    # Construct the polynomial
    poly = sum(coeff * s**(q_val + 1 - i) for i, coeff in enumerate(coefficients))

    # Compute Routh array
    pp = Poly(poly, s)
    routh_array = routh(pp)

    print()

    return {
        # 'polynomial': poly,
        'routh_array': routh_array,
        'first_column': routh_array[:, 0],
        # 'stability_conditions': sp.solve([e > 0 for e in routh_array[:, 0]], Q)
    }

# Example usage for q=2
q = 2
result = compute_routh_hurwitz(1)
# print("Polynomial:", result['polynomial'])
# print("Routh Array:", result['routh_array'])
print("First Column:", result['first_column'])
print("Stability Conditions:", result['stability_conditions'])
