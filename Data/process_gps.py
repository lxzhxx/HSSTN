import pandas as pd

file_path = r'grid_data.csv'
target = pd.read_csv(r'travel_all.csv')
data = pd.read_csv(file_path, header=None)

save_counter = 0

for index_1, row in target.iloc[:, [3, 4]].iterrows():
    a1, b1 = row
    second_column = data.iloc[:, 1]
    changed_rows = []
    changed_rows_index = []
    changed_rows.append(second_column.iloc[0])
    changed_rows_index.append(0)
    for index in range(1, len(second_column)):
        if second_column.iloc[index] != second_column.iloc[index - 1]:
            changed_rows.append(second_column.iloc[index])
            changed_rows_index.append(index)
    for index_lat, value_lat in zip(changed_rows_index, changed_rows):
        if a1 > value_lat:
            result_value_lat = changed_rows[changed_rows.index(value_lat) - 1]
            result_index_lat = changed_rows_index[changed_rows.index(value_lat) - 1]
            cnt = index_lat - result_index_lat
            break
    compare_value_lon = []
    compare_index_lon = []
    num1 = changed_rows_index[1]
    fourth_column_values = data.iloc[result_index_lat + 1:result_index_lat + cnt + 1, 3]

    compare_value_lon.extend(fourth_column_values)
    compare_index_lon.extend(range(result_index_lat + 1, result_index_lat + 1 + num1))
    for index_lon, value_lon in zip(compare_index_lon, compare_value_lon):
        if b1 < value_lon:
            result_value_lon = compare_value_lon[compare_value_lon.index((value_lon)) - 1]
            result_index_lon = compare_index_lon[compare_value_lon.index((value_lon)) - 1]
            break
    target.loc[index_1, 'end_code'] = result_index_lon

    save_counter += 1
    if save_counter % 1000 == 0:
        target.to_csv(r'data_new.csv')
        print(save_counter)
target.to_csv(r'data_new.csv')
