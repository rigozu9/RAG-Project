import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")
print(df["Region"].value_counts())

regional_summary = (
    df.groupby("Region")
    .agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        total_quantity=("Quantity", "sum"),
        transaction_count=("Order ID", "count")
    )
    .reset_index()
    .sort_values("total_sales", ascending=False)
)

print(regional_summary.head())

def regional_summary_to_text(row):
    return (
        f"Summary for region {row['Region']}: Total sales were ${row['total_sales']:.2f}, "
        f"total profit was ${row['total_profit']:.2f}, total quantity sold was {row['total_quantity']}, "
        f"and the number of transactions was {row['transaction_count']}."
    )

regional_summary["text"] = regional_summary.apply(regional_summary_to_text, axis=1)

print(regional_summary[["Region", "text"]])

regional_summary.to_csv("data/region_summaries.csv", index=False)