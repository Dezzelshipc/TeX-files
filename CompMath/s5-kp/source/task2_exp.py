import numpy as np
import matplotlib.pyplot as plt

size = int(input("Size of matrices: "))
tests = int(input("Number of tests: "))
rand = input("Number for alpha or 'r' for random numbers: ")

max_a = 100
min_a = 0.1

diff_list = []

for _ in range(tests):
    alpha = int(rand) if 'r' not in rand else (max_a - min_a) * np.random.random(1) + min_a
    rand_mat = np.random.random((size, size))

    cond = np.linalg.cond(rand_mat)
    cond_a = np.linalg.cond(alpha * rand_mat)

    diff_list.append(( abs( cond - cond_a ), cond, cond_a ))

diff_list = np.array(diff_list)

print("Max difference:", max(diff_list, key=lambda x: x[0]))
print("Max cond:", max(diff_list, key=lambda x: x[1]))
print("Min difference:", min(diff_list, key=lambda x: x[0]))
print("Min cond:", min(diff_list, key=lambda x: x[1]))
# print(diff_list)

# diff_list = np.round(diff_list, 7)

# try:
#     diff_list = np.ceil(np.log10(diff_list))
# except RuntimeWarning:
#     pass
# print("Max difference:", max(diff_list))

# unique, counts = np.unique(diff_list, return_counts=True)
# print(unique, counts)

# plt.bar(unique[1:], counts[1:], width=1, edgecolor='k', align='edge')
# plt.savefig(f'task3_{c}.pdf')
# plt.show()