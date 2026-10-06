"""`make run`: answer one example question end to end and print the result."""

import json
import os
from pathlib import Path

from rag_poc.answer import answer_question
from rag_poc.chunk import Chunk
from rag_poc.index import load_index
from rag_poc.retrieve import embed_query, top_k

ROOT = Path(__file__).resolve().parent.parent.parent
SNAPSHOT_PATH = ROOT / "data" / "snapshot.json"
INDEX_PATH = ROOT / "index" / "embeddings.json"

EXAMPLE_QUESTION = "What must my landlord do if my property needs repairs or has mould?"


def main() -> None:
    cheap_model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")

    records = json.loads(SNAPSHOT_PATH.read_text())
    chunks_by_id = {r["id"]: Chunk(**{k: v for k, v in r.items() if k != "fetch_date"}) for r in records}
    entries = load_index(INDEX_PATH)

    print(f"Question: {EXAMPLE_QUESTION}\n")
    query_vector = embed_query(EXAMPLE_QUESTION, embed_model=embed_model)
    retrieved = top_k(query_vector, entries, chunks_by_id, k=5)

    print("Top retrieved chunks:")
    for chunk, score in retrieved:
        print(f"  {score:.3f}  {chunk.heading_path}  [{chunk.id}]")

    result, llm_response = answer_question(
        question=EXAMPLE_QUESTION, retrieved=retrieved, cheap_model=cheap_model, similarity_threshold=0.3
    )

    print(f"\nAnswer: {result.answer}")
    print(f"Citations: {result.citations}")
    print(f"Refused: {result.refused}")
    if llm_response:
        print(
            f"Tokens: {llm_response.prompt_tokens} prompt / {llm_response.completion_tokens} "
            f"completion, {llm_response.latency_ms:.0f}ms"
        )


if __name__ == "__main__":
    main()
