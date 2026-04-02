import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")
category_df = pd.read_csv("data/category_summaries.csv")
region_df = pd.read_csv("data/region_summaries.csv")
monthly_df = pd.read_csv("data/monthly_summaries.csv")

# ---------Category Summary---------
best_category_sales = category_df.loc[category_df["total_sales"].idxmax()]
worst_category_sales = category_df.loc[category_df["total_sales"].idxmin()]

best_category_profit = category_df.loc[category_df["total_profit"].idxmax()]
worst_category_profit = category_df.loc[category_df["total_profit"].idxmin()]

# --------- Region Summary---------
best_region_sales = region_df.loc[region_df["total_sales"].idxmax()]
worst_region_sales = region_df.loc[region_df["total_sales"].idxmin()]

best_region_profit = region_df.loc[region_df["total_profit"].idxmax()]
worst_region_profit = region_df.loc[region_df["total_profit"].idxmin()]

# ---------- Monthly Summary---------
best_month_sales = monthly_df.loc[monthly_df["total_sales"].idxmax()]
worst_month_sales = monthly_df.loc[monthly_df["total_sales"].idxmin()]

best_month_profit = monthly_df.loc[monthly_df["total_profit"].idxmax()]
worst_month_profit = monthly_df.loc[monthly_df["total_profit"].idxmin()]

insights = f"""
Sales insights:

The highest-selling month was {best_month_sales['YearMonth']}, with total sales of ${best_month_sales['total_sales']:.2f}. 
The lowest-selling month was {worst_month_sales['YearMonth']}, with total sales of ${worst_month_sales['total_sales']:.2f}.

The most profitable month was {best_month_profit['YearMonth']}, generating ${best_month_profit['total_profit']:.2f} in profit. 
The least profitable month was {worst_month_profit['YearMonth']}, with profit of ${worst_month_profit['total_profit']:.2f}.

Among product categories, {best_category_sales['Category']} generated the highest sales (${best_category_sales['total_sales']:.2f}), 
while {worst_category_sales['Category']} had the lowest sales (${worst_category_sales['total_sales']:.2f}).

In terms of profitability, {best_category_profit['Category']} was the most profitable category (${best_category_profit['total_profit']:.2f}), 
whereas {worst_category_profit['Category']} was the least profitable (${worst_category_profit['total_profit']:.2f}).

Looking at regions, {best_region_sales['Region']} achieved the highest total sales (${best_region_sales['total_sales']:.2f}), 
while {worst_region_sales['Region']} had the lowest sales (${worst_region_sales['total_sales']:.2f}).

In terms of profit, {best_region_profit['Region']} was the most profitable region (${best_region_profit['total_profit']:.2f}), 
while {worst_region_profit['Region']} had the lowest profit (${worst_region_profit['total_profit']:.2f}).

Overall, {best_category_sales['Category']} and the {best_region_sales['Region']} region show the strongest performance in this dataset.
"""

stats_text = f"""
Overall sales statistics:

There were {len(df)} transactions in total. The average sales per transaction were ${df['Sales'].mean():.2f}, 
with a maximum sale of ${df['Sales'].max():.2f} and a minimum sale of ${df['Sales'].min():.2f}.

The average profit was ${df['Profit'].mean():.2f}. The highest profit recorded was ${df['Profit'].max():.2f}, 
while the lowest profit was ${df['Profit'].min():.2f}.

{insights}
"""

with open("data/statistical_summary.txt", "w") as f:
    f.write(stats_text)