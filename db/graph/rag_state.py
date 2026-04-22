from typing import TypedDict

class RAGState(TypedDict, total=False):
    query: str
    filters: dict
    context: str
    answer: str
    messages: list
