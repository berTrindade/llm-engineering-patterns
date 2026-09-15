"""Chunk embeddings -> a flat JSON index (gitignored, rebuilt by `make index`)."""

import json
from pathlib import Path

from pydantic import BaseModel

from rag_poc.chunk import Chunk
from rag_poc.llm import embed

DOCUMENT_PREFIX = "search_document: "


class IndexEntry(BaseModel):
    id: str
    vector: list[float]


def build_index(chunks: list[Chunk], embed_model: str) -> list[IndexEntry]:
    texts = [DOCUMENT_PREFIX + chunk.text for chunk in chunks]
    vectors = embed(texts, model=embed_model)
    return [IndexEntry(id=chunk.id, vector=vector) for chunk, vector in zip(chunks, vectors, strict=True)]


def save_index(entries: list[IndexEntry], path: Path) -> None:
    path.write_text(json.dumps([e.model_dump() for e in entries]))


def load_index(path: Path) -> list[IndexEntry]:
    return [IndexEntry(**item) for item in json.loads(path.read_text())]
