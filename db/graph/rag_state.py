from typing import TypedDict

class RAGState(TypedDict, total=False):
    query: str
    filters: dict
    previous_filters: dict
    context: str
    answer: str
    messages: list
