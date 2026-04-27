from db.search_functions import summary_similarity_search

query = "Which region has the highest sales?"

matches = summary_similarity_search(query, top_k=3)

for rank, match in enumerate(matches, start=1):
    print(f"\nResult {rank}")
    print("ID:", match["id"])
    print("Distance:", match["distance"])
    print("Metadata:", match["metadata"])
    print("Document:", match["document"])
