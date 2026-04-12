import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_FILE = "data/chunks_500.csv"
SUPERSTORE_FILE = "data/superstore.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_chunks_500"
BATCH_SIZE = 200

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

ids = [f"chunk_{chunk_id}" for chunk_id in df_merged["chunk_id"]]
documents = df_merged["text"].tolist()

metadatas = []
for _, row in df_merged.iterrows():
    metadata = {
        "row_id": int(row["Row ID"]),
        "order_id": str(row["Order ID"]),
        "order_date": str(row["Order Date"]),
        "ship_date": str(row["Ship Date"]),
        "segment": str(row["Segment"]),
        "city": str(row["City"]),
        "state": str(row["State"]),
        "region": str(row["Region"]),
        "category": str(row["Category"]),
        "sub_category": str(row["Sub-Category"]),
        "sales": float(row["Sales"]),
        "quantity": int(row["Quantity"]),
        "discount": float(row["Discount"]),
        "profit": float(row["Profit"])
    }
    metadatas.append(metadata)

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.get_or_create_collection(name=COLLECTION_NAME)

for i in range(0, len(ids), BATCH_SIZE):
    batch_ids = ids[i:i + BATCH_SIZE]
    batch_documents = documents[i:i + BATCH_SIZE]
    batch_embeddings = embeddings[i:i + BATCH_SIZE]
    batch_metadatas = metadatas[i:i + BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        documents=batch_documents,
        embeddings=batch_embeddings.tolist(),
        metadatas=batch_metadatas
    )

    print(f"Stored batch {i} to {i + len(batch_ids) - 1}")

print("Embeddings and metadata stored successfully in ChromaDB.")
print("Collection name:", COLLECTION_NAME)
print("Total records:", collection.count())
