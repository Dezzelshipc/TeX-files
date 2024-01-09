import numpy as np


def main_solve(solve_func, matrix=None, values=None):
    if matrix is None or values is None:
        print("Matrix or values not present")

    my_sol = solve_func(matrix, values)
    np_sol = np.linalg.solve(matrix, values)
    diff = my_sol - np_sol

    print(f"my_sol:\n{list(my_sol)}\n", f"max diff for my_sol - np_sol = {np.max(abs(diff))}",
          sep='\n')
    
    my_f = matrix.dot(my_sol)
    print(f"max diff for my_f - f = {np.max(abs(values-my_f))}")
    print(np.linalg.cond(matrix), "&", np.max(abs(diff)), "&", np.max(abs(values-my_f)))
    

def read_data(file_name: str) -> (np.matrix, np.ndarray):
    with open(file_name, "r") as f:
        start = f.readline().split()
        raw_mat = [start]
        for _ in range(len(start) - 1):
            raw_mat.append(f.readline().split())

        s = f.read().strip()

        return np.matrix(raw_mat).astype(float), np.array(s.split()).astype(float)
