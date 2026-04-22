import pandas as pd

from db.graph.rag_graph import build_rag_graph
from db.search_functions import get_unique_metadata_values

SUPERSTORE_FILE = "data/superstore.csv"

def load_metadata_values():
    df_superstore = pd.read_csv(SUPERSTORE_FILE, encoding="latin1")

    regions = get_unique_metadata_values(df_superstore, "Region")
    categories = get_unique_metadata_values(df_superstore, "Category")
    segments = get_unique_metadata_values(df_superstore, "Segment")

    return regions, categories, segments

def main():
    regions, categories, segments = load_metadata_values()

    graph = build_rag_graph(
        regions,
        categories,
        segments
    )

    config = {
        "configurable": {
            "thread_id": "chat_1"
        }
    }

    print("RAG chat started. Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        result = graph.invoke(
            {
                "query": query
            },
            config=config
        )

        print("\nAssistant:")
        print(result["answer"])

if __name__ == "__main__":
    main()