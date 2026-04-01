import pandas as pd

df = pd.read_csv("data/superstore.csv", encoding="latin1")

def row_to_text(row):
    return (
        f"Transaction details: Row ID {row['Row ID']}, Order ID {row['Order ID']}. "
        f"Order date: {row['Order Date']}, Ship date: {row['Ship Date']}, Ship mode: {row['Ship Mode']}. "
        f"Customer: {row['Customer Name']} (Customer ID: {row['Customer ID']}), Segment: {row['Segment']}. "
        f"Location: {row['City']}, {row['State']}, Postal Code {row['Postal Code']}, "
        f"{row['Country']} in the {row['Region']} region. "
        f"Product: {row['Product Name']} (Product ID: {row['Product ID']}), "
        f"Category: {row['Category']}, Sub-category: {row['Sub-Category']}. "
        f"Quantity: {row['Quantity']}, Sales: ${row['Sales']:.2f}, "
        f"Discount: {row['Discount']:.2f}, Profit: ${row['Profit']:.2f}."
    )

# Add order id aswell as text to easier link to og data if needed
df_text = df[["Order ID"]].copy()
df_text["text"] = df.apply(row_to_text, axis=1)

# df_text.to_csv("data/transactions_with_text.csv", index=False)

print(df_text.head())