from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from ollama import chat

from db.graph.rag_state import RAGState
from db.search_functions import similarity_search, extract_metadata_filters

OLLAMA_MODEL = "llama3:latest"

def build_rag_graph(regions, categories, segments):
    def extract_filters_node(state):
        query = state["query"]

        filters = extract_metadata_filters(
            query,
            regions,
            categories,
            segments
        )

        return {
            "filters": filters
        }

    def retrieve_context_node(state):
        query = state["query"]
        filters = state.get("filters", {})

        matches = similarity_search(
            query,
            top_k=5,
            filters=filters
        )

        context = ""

        for match in matches:
            context += match["document"] + "\n\n"

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
            Use only the context below to answer the user's question.
            Use the conversation history only to understand follow-up questions.
            If the context does not contain enough information, say so.

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
