import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")
print(df.isnull().sum()) # 0 in all rows

# Check dtypes
print(df.dtypes)

# Check duplicate rows
print("Duplicate rows:", df.duplicated().sum()) # 0