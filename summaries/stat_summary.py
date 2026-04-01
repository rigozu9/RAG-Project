import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")

stats_text = f"""
Overall sales statistics:
Total transactions: {len(df)}
Average sales: ${df['Sales'].mean():.2f}
Maximum sale: ${df['Sales'].max():.2f}
Minimum sale: ${df['Sales'].min():.2f}

Average profit: ${df['Profit'].mean():.2f}
Maximum profit: ${df['Profit'].max():.2f}
Minimum profit: ${df['Profit'].min():.2f}
"""

print(stats_text)

with open("data/statistical_summary.txt", "w") as f:
    f.write(stats_text)