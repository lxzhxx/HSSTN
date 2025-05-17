import pandas as pd

file_path = r'all_data.csv'
output_path = r'data.csv'

df = pd.read_csv(file_path, header=None)

unique_values = sorted(set(df.iloc[:, 1]).union(df.iloc[:, 2]))

unique_values_df = pd.DataFrame(unique_values, columns=['Sorted Unique Values'])
unique_values_df.to_csv(output_path, index=False, encoding='utf-8')

print(output_path)
