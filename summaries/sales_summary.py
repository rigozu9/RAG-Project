import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")

df["Order Date"] = pd.to_datetime(df["Order Date"])
# print(df["Order Date"].head())

df["YearMonth"] = df["Order Date"].dt.to_period("M")
# print(df[["Order Date", "YearMonth"]].head())

monthly_summary = (
    df.groupby("YearMonth")
    .agg(
        total_sales=("Sales", "sum"),
        total_profit=("Profit", "sum"),
        total_quantity=("Quantity", "sum"),
        transaction_count=("Order ID", "count")
    )
    .reset_index()
    .sort_values("YearMonth")
)

# print(monthly_summary.head())

def monthly_summary_to_text(row):
    return (
        f"Summary for {row['YearMonth']}: Total Sales: ${row['total_sales']:.2f}, "
        f"Total Profit: ${row['total_profit']:.2f}, Total Quantity Sold: {row['total_quantity']}, "
        f"Number of Transactions: {row['transaction_count']}."
    )

monthly_summary["text"] = monthly_summary.apply(monthly_summary_to_text, axis=1)

print(monthly_summary[["YearMonth", "text"]].head())

# monthly_summary.to_csv("data/monthly_summaries.csv", index=False)