from db.search_functions import combined_similarity_search

query = "Which region has the highest sales?"

results = combined_similarity_search(
    query,
    transaction_top_k=3,
    summary_top_k=3
)

print("\nSUMMARY RESULTS")
for rank, match in enumerate(results["summaries"], start=1):
    print(f"\nSummary result {rank}")
    print("ID:", match["id"])
    print("Distance:", match["distance"])
    print("Metadata:", match["metadata"])
    print("Document:", match["document"])

print("\nTRANSACTION RESULTS")
for rank, match in enumerate(results["transactions"], start=1):
    print(f"\nTransaction result {rank}")
    print("ID:", match["id"])
    print("Distance:", match["distance"])
    print("Metadata:", match["metadata"])
    print("Document:", match["document"])
