import numpy as np

size = int(input("Size of matrices: "))
tests = int(input("Number of tests: "))
3

norm_list = []

for _ in range(tests):
    rand_mat = np.random.random((size, size))

    inf_norm = np.linalg.norm(rand_mat, ord=np.inf)
    m_norm = size * np.max(rand_mat)

    norm_list.append(( inf_norm, m_norm, size * inf_norm, 
                      inf_norm <= m_norm <= size * inf_norm ))


print(*norm_list, sep="\n")
a,a,a, tr = zip(*norm_list)
tr = np.array(tr)
print(tr.all())
