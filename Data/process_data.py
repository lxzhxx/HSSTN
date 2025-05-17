import numpy as np

file_path = r'matrix_huizhou.npy'
data = np.load(file_path)

matrix_value_list = [0]

cumulative_sum = 0
for matrix in data:
    matrix_sum = np.sum(matrix)
    cumulative_sum += matrix_sum
    matrix_value_list.append(cumulative_sum)

matrix_value_list = [int(value) for value in matrix_value_list]

print(matrix_value_list[-1])

npy_file_path = r'point_huizhou.npy'
np.save(npy_file_path, matrix_value_list)

print(npy_file_path)

