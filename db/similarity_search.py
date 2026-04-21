import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from ollama import chat

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_chunks_500"
MODEL_NAME = "all-MiniLM-L6-v2"
SUPERSTORE_FILE = "data/superstore.csv"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
df_superstore = pd.read_csv(SUPERSTORE_FILE, encoding="latin1")

def get_unique_metadata_values(df, column_name):
    values = df[column_name].dropna().unique().tolist()
    return values

def extract_metadata_filters(query, regions, categories, segments):
    query_lower = query.lower()

    filters = {}

    for region in regions:
        if region.lower() in query_lower:
            filters["region"] = region

    for category in categories:
        if category.lower() in query_lower:
            filters["category"] = category

    for segment in segments:
        if segment.lower() in query_lower:
            filters["segment"] = segment

    return filters

def build_where_filter(filters):
    if not filters:
        return None

    conditions = []

    for key, value in filters.items():
        conditions.append({key: value})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}

def similarity_search(query, top_k, filters=None):
    query_embedding = SentenceTransformer(MODEL_NAME).encode(query).tolist()

    where_filter = build_where_filter(filters)
    print("Chroma where filter:", where_filter)

    if where_filter:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

    matches = []

    for i in range(len(results["ids"][0])):
        matches.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return matches

query = "show me technology sales in the east region for consumer segment"

regions = get_unique_metadata_values(df_superstore, "Region")
categories = get_unique_metadata_values(df_superstore, "Category")
segments = get_unique_metadata_values(df_superstore, "Segment")

filters = extract_metadata_filters(
    query,
    regions,
    categories,
    segments
)

print(filters)

matches = similarity_search(query, top_k=5, filters=filters)

context = ""

for match in matches:
    context += match["document"] + "\n\n"

prompt = f"""
Role: You are a sales data analyst. 

Instructions:
Use only the context below to answer the user's question.
If the context does not contain enough information, say so.

Context:
{context}

Based on the context, answer the query.
Query:
{query}

Answer:
"""

response = chat(
    model="llama3:latest",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print(response["message"]["content"])
