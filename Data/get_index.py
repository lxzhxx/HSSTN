import pandas as pd

file_path = r'part_all.csv'
output_path = r'part_all_new.csv'

df = pd.read_csv(file_path, header=None, usecols=[3])
unique_values = df.iloc[:, 0].unique()
value_to_row = {value: idx + 1 for idx, value in enumerate(unique_values)}  # 从1开始编号

with open(output_path, 'w', encoding='utf-8') as f:
    pass

chunk_size = 5000
for chunk in pd.read_csv(file_path, header=None, chunksize=chunk_size):

    chunk['Row_2_in_5'] = chunk.iloc[:, 1].map(value_to_row).fillna('Not found').astype(int)
    chunk['Row_3_in_5'] = chunk.iloc[:, 2].map(value_to_row).fillna('Not found').astype(int)


    chunk.to_csv(output_path, mode='a', index=False, header=False, encoding='utf-8')

    print(len(chunk))

print(output_path)
