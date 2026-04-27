from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from ollama import chat

from db.graph.rag_state import RAGState
from db.search_functions import combined_similarity_search, extract_metadata_filters

OLLAMA_MODEL = "llama3:latest"

def build_rag_graph(regions, categories, segments):
    def extract_filters_node(state):
        query = state["query"]
        previous_filters = state.get("previous_filters", {})
        current_filters = extract_metadata_filters(
            query,
            regions,
            categories,
            segments
        )

        merged_filters = previous_filters.copy()
        merged_filters.update(current_filters)
        print("Current filters:", current_filters)
        print("Merged filters:", merged_filters)

        return {
            "filters": merged_filters,
            "previous_filters": merged_filters
        }

    def retrieve_context_node(state):
        query = state["query"]
        filters = state.get("filters", {})

        matches = combined_similarity_search(
            query,
            transaction_top_k=5,
            summary_top_k=5,
            filters=filters
        )

        context = "Aggregate summaries:\n"

        for match in matches["summaries"]:
            context += match["document"] + "\n"

        context += "\nTransaction examples:\n"

        for match in matches["transactions"]:
            context += match["document"] + "\n"

        return {
            "context": context
        }

    def generate_answer_node(state):
        query = state["query"]
        context = state["context"]
        messages = state.get("messages", [])

        prompt = f"""
            Role: You are a sales data analyst.
            
            Instructions:
            Answer only the current user question.
            Use the conversation history only to resolve references like "that region", "that category", or "those years".
            Do not repeat a previous answer unless the current user question asks for it.
            Use only the context below as factual evidence.
            If the context does not contain enough information, say so clearly.

            Context rules:
            - Use aggregate summaries for overall totals, comparisons, rankings, and trends.
            - Use transaction examples only as supporting examples or when the user asks about specific orders.
            - Do not calculate totals from transaction examples if aggregate summaries are available.
            - If aggregate summaries and transaction examples conflict, prefer the aggregate summaries.
            Context:
            {context}
            Current user question:
            {query}
            Answer:
        """

        ollama_messages = messages + [
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = chat(
            model=OLLAMA_MODEL,
            messages=ollama_messages
        )

        answer = response["message"]["content"]

        updated_messages = messages + [
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]

        return {
            "answer": answer,
            "messages": updated_messages
        }

    workflow = StateGraph(RAGState)

    workflow.add_node("extract_filters", extract_filters_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.add_edge(START, "extract_filters")
    workflow.add_edge("extract_filters", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)
