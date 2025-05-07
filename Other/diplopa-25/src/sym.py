from sympy import Eq, Symbol, IndexedBase, linsolve, nonlinsolve, factor, simplify, solve

def solve_recursive_system(s, q, r):
    # Define IndexedBases for variables
    N = IndexedBase('N')
    N_prime = IndexedBase('N_prime')
    
    # Define constants
    Q = Symbol('Q')
    a1 = Symbol('a1')
    N0 = Symbol('N0')
    N1 = Symbol("N1")
    alpha = IndexedBase('alpha')
    alpha0 = alpha[0]
    k = IndexedBase('k')
    m = IndexedBase('m')
    alpha_s_b = Symbol('alpha_s^b')
    alpha_prime = IndexedBase('alpha_prime')
    k_prime = IndexedBase('k_prime')
    m_prime = IndexedBase('m_prime')
    
    equations = []
    
    # First equation: N1 = Q / (a1*m1 - alpha0*N0)
    equations.append(Eq(N[1], N1))
    
    # Main chain equations for i=1 to s-1 and s+1 to q
    for i in list(range(1, s)) + list(range(s+1, q+1)):
        if i == 1:
            rhs = k[i] * alpha0 * N0 - m[i]
        else:
            rhs = k[i] * alpha[i-1] * N[i-1] - m[i]
        equations.append(Eq(alpha[i] * N[i+1] if i < q else 0, rhs))
    
    # Equation for i=s
    if s >= 1:
        if s == 1:
            rhs_term = k[s] * alpha0 * N0
        else:
            rhs_term = k[s] * alpha[s-1] * N[s-1]
        rhs = rhs_term - m[s]
        lhs = alpha_s_b * N_prime[1] + alpha[s] * N[s+1]
        equations.append(Eq(lhs, rhs))
    
    # Primed equations
    # k=1
    equations.append(Eq(alpha_prime[1] * N_prime[2], k_prime[1] * alpha_s_b * N[s] - m_prime[1]))
    
    # k=2 to r
    for k_val in range(2, r+1):
        lhs = alpha_prime[k_val] * N_prime[k_val + 1] if k_val < r else 0
        rhs = k_prime[k_val] * alpha_prime[k_val - 1] * N_prime[k_val - 1] - m_prime[k_val]
        equations.append(Eq(lhs, rhs))

    # equations.extend([
    #     Eq(N_prime[r+1], 0)
    #     ])

    [print(eq, end="\n\n") for eq in equations]
    
    # Collect variables
    variables = [N[i] for i in range(1, q+2)] + [N_prime[k] for k in range(1, r+2)]
    
    # Solve the system
    solution = solve(equations, variables)
    return solution

# Example usage:
q = 6
r = 3
solution = solve_recursive_system(s=2, q=q, r=r)
print("-----------\n")
# print(solution)
for sol in solution:
    for V, S in zip(list(range(1, q+2)) + list(range(1,r+2)), sol , strict=True):
        print(V, S, "\n")
# [print(factor(s), "\n") for sol in solution for s in sol]