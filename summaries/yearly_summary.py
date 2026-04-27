import pandas as pd

DATA_FILE = "data/superstore.csv"
OUTPUT_FILE = "data/yearly_summaries.csv"

df = pd.read_csv(DATA_FILE, encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"] = df["Order Date"].dt.year

yearly_summary = (
    df.groupby("Year")
    .agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        total_quantity=("Quantity", "sum"),
        transaction_count=("Row ID", "count")
    )
    .reset_index()
)

yearly_summary["text"] = yearly_summary.apply(
    lambda row: (
        f"Summary for year {int(row['Year'])}: "
        f"Total sales were ${row['total_sales']:.2f}, "
        f"total profit was ${row['total_profit']:.2f}, "
        f"total quantity sold was {int(row['total_quantity'])}, "
        f"and the number of transactions was {int(row['transaction_count'])}."
    ),
    axis=1
)

yearly_summary.to_csv(OUTPUT_FILE, index=False)
print("Yearly summaries saved to:", OUTPUT_FILE)
print(yearly_summary)
