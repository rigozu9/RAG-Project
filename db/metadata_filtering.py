import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_chunks_500"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

def metadata_filter_search(metadata_filter, limit):
    results = collection.get(
        where=metadata_filter,
        limit=limit,
        include=["documents", "metadatas"]
    )

    matches = []
    for i in range(len(results["ids"])):
        matches.append({
            "id": results["ids"][i],
            "document": results["documents"][i],
            "metadata": results["metadatas"][i]
        })

    return matches

matches = metadata_filter_search(
    {
        "$and": [
            {"region": "East"},
            {"category": "Furniture"},
            {"sub_category": "Chairs"}
        ]
    },
    limit=5
)

for rank, match in enumerate(matches, start=1):
    print(f"Result {rank}")
    print("ID:", match["id"])
    print("Metadata:", match["metadata"])
    print("Document:", match["document"][:300], "...\n")
