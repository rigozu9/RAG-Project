import chromadb

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="sales_chunks"
)

print("ChromaDB is ready.")
print("Collection name:", collection.name)