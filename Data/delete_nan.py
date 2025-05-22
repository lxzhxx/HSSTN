import pandas as pd

file_path = r"huizhou.csv"
data = pd.read_csv(file_path)

data.dropna(inplace=True)

cleaned_file_path = r"huizhou_cleaned.csv"
data.to_csv(cleaned_file_path, index=False)

