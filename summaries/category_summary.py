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

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Year"] = df["Order Date"].dt.year

category_year_sales = (
    df.groupby(["Category", "Year"])
    .agg(year_sales=("Sales", "sum"))
    .reset_index()
)

best_years = (
    category_year_sales.sort_values(["Category", "year_sales"], ascending=[True, False])
    .groupby("Category")
    .first()
    .reset_index()
    .rename(columns={
        "Year": "best_sales_year",
        "year_sales": "best_year_sales"
    })
)

worst_years = (
    category_year_sales.sort_values(["Category", "year_sales"], ascending=[True, True])
    .groupby("Category")
    .first()
    .reset_index()
    .rename(columns={
        "Year": "worst_sales_year",
        "year_sales": "worst_year_sales"
    })
)

category_summary = category_summary.merge(
    best_years,
    on="Category",
    how="left"
)

category_summary = category_summary.merge(
    worst_years,
    on="Category",
    how="left"
)

print(category_summary.head())

def category_summary_to_text(row):
    return (
        f"Summary for category {row['Category']}: Total sales were ${row['total_sales']:.2f}, "
        f"total profit was ${row['total_profit']:.2f}, total quantity sold was {int(row['total_quantity'])}, "
        f"and the number of transactions was {int(row['transaction_count'])}. "
        f"The best sales year for this category was {int(row['best_sales_year'])} with ${row['best_year_sales']:.2f} in sales, "
        f"and the worst sales year was {int(row['worst_sales_year'])} with ${row['worst_year_sales']:.2f} in sales."
    )

category_summary["text"] = category_summary.apply(category_summary_to_text, axis=1)
print(category_summary[["Category", "text"]].head())

category_summary.to_csv("data/category_summaries.csv", index=False)