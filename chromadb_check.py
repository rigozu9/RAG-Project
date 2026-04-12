import chromadb

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "sales_chunks_500"

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

print("Collection count:", collection.count())

result = collection.get(
    ids=["chunk_1"],
    include=["documents", "metadatas", "embeddings"]
)

print("\nStored record preview:")
print("ID:", result["ids"][0])
print("Document:", result["documents"][0][:300], "...")
print("Metadata:", result["metadatas"][0])
print("Embedding length:", len(result["embeddings"][0]))
