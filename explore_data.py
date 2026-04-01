import pandas as pd

df = pd.read_csv("superstore.csv", encoding="latin1")
print(df.head())
print(df.iloc[0])
print(df.columns)