import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")
# print(df["Category"].value_counts())

category_summary = (
    df.groupby("Category")
    .agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        total_quantity=("Quantity", "sum"),
        transaction_count=("Order ID", "count")
    )
    .reset_index()
    .sort_values("total_sales", ascending=False)
)

print(category_summary.head())

def category_summary_to_text(row):
    return (
        f"Summary for category {row['Category']}: Total sales were ${row['total_sales']:.2f}, "
        f"total profit was ${row['total_profit']:.2f}, total quantity sold was {row['total_quantity']}, "
        f"and the number of transactions was {row['transaction_count']}."
    )

category_summary["text"] = category_summary.apply(category_summary_to_text, axis=1)
print(category_summary[["Category", "text"]].head())

category_summary.to_csv("data/category_summaries.csv", index=False)