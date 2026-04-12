import pandas as pd
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/chunks_500.csv"
SUPERSTORE_FILE = "data/superstore.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

df_chunks = pd.read_csv(CHUNK_FILE)
df_superstore = pd.read_csv(SUPERSTORE_FILE, encoding="latin1")

metadata_columns = [
    "Row ID",
    "Order ID",
    "Order Date",
    "Ship Date",
    "Segment",
    "City",
    "State",
    "Region",
    "Category",
    "Sub-Category",
    "Sales",
    "Quantity",
    "Discount",
    "Profit"
]

df_metadata = df_superstore[metadata_columns].copy()

df_merged = df_chunks.merge(
    df_metadata,
    left_on="row_id",
    right_on="Row ID",
    how="left"
)

df_merged["text"] = df_merged["text"].astype(str).str.strip()
df_merged = df_merged[df_merged["text"] != ""].copy()

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    df_merged["text"].tolist(),
    show_progress_bar=True
)

df_merged["embedding"] = list(embeddings)

print("Embeddings generated successfully.")
print("Rows:", len(df_merged))
print("Embedding dimension:", len(df_merged["embedding"].iloc[0]))

print("\nPreview:")
print(df_merged[["chunk_id", "row_id", "Order ID", "Category", "Region"]].head())
