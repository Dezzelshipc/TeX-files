from sympy import Eq, Symbol, IndexedBase, linsolve, nonlinsolve

def solve_recursive_system(q):
    # Define IndexedBases for variables
    N = IndexedBase('N')
    
    # Define constants
    Q = Symbol('Q')
    N0 = Symbol('N0')
    N1 = Symbol('N1')
    alpha = IndexedBase('alpha')
    alpha0 = alpha[0]
    k = IndexedBase('k')
    m = IndexedBase('m')
    
    equations = []
    
    # First equation: N1 = Q / (a1*m1 - alpha0*N0)
    equations.append(Eq(N[1], Q / (alpha0 * N0)))
    
    # Main chain equations for i=1 to s-1 and s+1 to q
    for i in list(range(1, q+1)):
        if i == 1:
            rhs = k[i] * alpha0 * N0 - m[i]
        else:
            rhs = k[i] * alpha[i-1] * N[i-1] - m[i]
        equations.append(Eq(alpha[i] * N[i+1], rhs))

    # equations.extend([
    #     Eq(N[q+1], 0)
    #     ])

    [print(eq, end="\n\n") for eq in equations]
    
    # Collect variables
    variables = [N[i] for i in range(1,q+2)]
    print(variables)
    
    # Solve the system
    solution = nonlinsolve(equations, variables)
    return solution

# Example usage:
solution = solve_recursive_system(4)

print("-----------\n")
print(solution)
[print(s, "\n") for sol in solution for s in sol]