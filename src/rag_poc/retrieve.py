"""Cosine similarity search over the JSON index, pure Python (fine at this corpus size)."""

import math

from rag_poc.chunk import Chunk
from rag_poc.index import IndexEntry
from rag_poc.llm import embed

QUERY_PREFIX = "search_query: "


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def embed_query(question: str, embed_model: str) -> list[float]:
    return embed([QUERY_PREFIX + question], model=embed_model)[0]


def top_k(
    query_vector: list[float],
    entries: list[IndexEntry],
    chunks_by_id: dict[str, Chunk],
    k: int = 5,
) -> list[tuple[Chunk, float]]:
    scored = [
        (chunks_by_id[entry.id], cosine_similarity(query_vector, entry.vector))
        for entry in entries
        if entry.id in chunks_by_id
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]
