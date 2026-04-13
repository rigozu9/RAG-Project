import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_chunks_500"
MODEL_NAME = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

def similarity_search(query, top_k):
    query_embedding = SentenceTransformer(MODEL_NAME).encode(query).tolist()

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

query = "technology sales in the east region"

for rank, match in enumerate(similarity_search(query, top_k=3), start=1):
    print(f"Result {rank}")
    print("ID:", match["id"])
    print("Distance:", match["distance"])
    print("Metadata:", match["metadata"])
    print("Document:", match["document"][:300], "...\n")
