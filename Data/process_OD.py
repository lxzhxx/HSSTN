import numpy as np
import pandas as pd
from tqdm import trange

od_mats = []
heads = []
tail1s = []
tail2s = []

input_len = 3600
output_len = 3600
n_nodes = 279


def build_od_matrix(source, destination, n_nodes):
    od_matrix = np.zeros(shape=[n_nodes, n_nodes])
    mylen = len(source)
    for i in range(mylen):
        # print(source[i])
        # print(destination[i])
        # print('-'*40)
        od_matrix[int(source[i])][int(destination[i])] += 1
    return od_matrix


def copy(a):
    if isinstance(a, list):
        return [copy(b) for b in a]
    else:
        return a


train_data = pd.read_csv(r"huizhou.csv")
end_time = 7559280

head = 0
tail1 = 0
tail2 = 0
train_data_size = len(train_data.iloc[:, 2])
empty_flag = 1
od_tim = [np.zeros([n_nodes, n_nodes])]
num_batch = (end_time - input_len) // output_len
batch_range = trange(num_batch)

for j in batch_range:
    st1 = j * output_len
    ed1 = j * output_len + input_len
    ed2 = (j + 1) * output_len + input_len
    now = copy(od_tim[-1])
    while head < train_data_size and train_data.iloc[:, 2][head] < st1:
        head += 1
    while tail1 < train_data_size and train_data.iloc[:, 2][tail1] < ed1:
        now[int(train_data.iloc[:, 1][tail1])][int(train_data.iloc[:, 1][tail1])] = train_data.iloc[:, 2][tail1]

        # now[train_data.iloc[:, 1][tail1]][train_data.iloc[:, 2][tail1]] = train_data.iloc[:, 3][tail1]

        tail1 += 1
    while tail2 < train_data_size and train_data.iloc[:, 2][tail2] < ed2:
        tail2 += 1

    heads.append(head)
    tail1s.append(tail1)
    tail2s.append(tail2)
    od_tim.append(now)

    od_matrix_real = build_od_matrix(train_data.iloc[:, 0][tail1:tail2].tolist(),
                                     train_data.iloc[:, 1][tail1:tail2].tolist(), n_nodes)

    od_mats.append(od_matrix_real)

npy_file = r'matrix_huizhou.npy'
np.save(npy_file, od_mats)
