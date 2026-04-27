import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_summaries"
BATCH_SIZE = 50

SUMMARY_FILES = [
    {
        "path": "data/category_summaries.csv",
        "summary_type": "category",
        "key_column": "Category",
        "metadata_key": "category"
    },
    {
        "path": "data/region_summaries.csv",
        "summary_type": "region",
        "key_column": "Region",
        "metadata_key": "region"
    },
    {
        "path": "data/monthly_summaries.csv",
        "summary_type": "monthly",
        "key_column": "YearMonth",
        "metadata_key": "year_month"
    }
]

STATISTICAL_SUMMARY_FILE = "data/statistical_summary.txt"

def load_summary_rows():
    summary_rows = []

    for summary_file in SUMMARY_FILES:
        df = pd.read_csv(summary_file["path"])

        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""].copy()

        for _, row in df.iterrows():
            metadata = {
                "summary_type": summary_file["summary_type"],
                "source_file": summary_file["path"],
                summary_file["metadata_key"]: str(row[summary_file["key_column"]]),
                "total_sales": float(row["total_sales"]),
                "total_profit": float(row["total_profit"]),
                "total_quantity": int(row["total_quantity"]),
                "transaction_count": int(row["transaction_count"])
            }

            if summary_file["summary_type"] == "monthly":
                metadata["year"] = str(row["YearMonth"]).split("-")[0]

            summary_rows.append({
                "id": f"{summary_file['summary_type']}_{row[summary_file['key_column']]}",
                "text": row["text"],
                "metadata": metadata
            })

    with open(STATISTICAL_SUMMARY_FILE, "r", encoding="utf-8") as file:
        statistical_text = file.read().strip()

    if statistical_text:
        summary_rows.append({
            "id": "statistical_summary",
            "text": statistical_text,
            "metadata": {
                "summary_type": "statistical",
                "source_file": STATISTICAL_SUMMARY_FILE
            }
        })

    return summary_rows

summary_rows = load_summary_rows()

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    [row["text"] for row in summary_rows],
    show_progress_bar=True
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.get_or_create_collection(name=COLLECTION_NAME)

for i in range(0, len(summary_rows), BATCH_SIZE):
    batch_rows = summary_rows[i:i + BATCH_SIZE]
    batch_embeddings = embeddings[i:i + BATCH_SIZE]

    collection.add(
        ids=[row["id"] for row in batch_rows],
        documents=[row["text"] for row in batch_rows],
        embeddings=batch_embeddings.tolist(),
        metadatas=[row["metadata"] for row in batch_rows]
    )

    print(f"Stored summary batch {i} to {i + len(batch_rows) - 1}")

print("Summary embeddings and metadata stored successfully in ChromaDB.")
print("Collection name:", COLLECTION_NAME)
print("Total records:", collection.count())
