import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_chunks_500"
SUMMARY_COLLECTION_NAME = "sales_summaries"
MODEL_NAME = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
summary_collection = client.get_collection(name=SUMMARY_COLLECTION_NAME)
model = SentenceTransformer(MODEL_NAME)

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
    query_embedding = model.encode(query).tolist()

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

def summary_similarity_search(query, top_k):
    query_embedding = model.encode(query).tolist()

    results = summary_collection.query(
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

def combined_similarity_search(query, transaction_top_k=5, summary_top_k=3, filters=None):
    summary_matches = summary_similarity_search(
        query,
        top_k=summary_top_k
    )

    transaction_matches = similarity_search(
        query,
        top_k=transaction_top_k,
        filters=filters
    )

    return {
        "summaries": summary_matches,
        "transactions": transaction_matches
    }
