import json

from rag_poc.chunk import Chunk
from rag_poc.index import IndexEntry, build_index, load_index, save_index


def test_save_and_load_index_round_trip(tmp_path):
    entries = [IndexEntry(id="a", vector=[0.1, 0.2]), IndexEntry(id="b", vector=[0.3, 0.4])]
    path = tmp_path / "embeddings.json"

    save_index(entries, path)
    loaded = load_index(path)

    assert loaded == entries
    assert json.loads(path.read_text())[0]["id"] == "a"


def test_build_index_prefixes_search_document_and_calls_embed(monkeypatch):
    seen = {}

    def fake_embed(texts, model):
        seen["texts"] = texts
        seen["model"] = model
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("rag_poc.index.embed", fake_embed)

    chunks = [
        Chunk(id="a", text="Fix the boiler.", source_url="u", page_title="p", heading_path="h"),
        Chunk(id="b", text="Protect the deposit.", source_url="u", page_title="p", heading_path="h"),
    ]

    entries = build_index(chunks, embed_model="nomic-embed-text")

    assert seen["texts"] == ["search_document: Fix the boiler.", "search_document: Protect the deposit."]
    assert seen["model"] == "nomic-embed-text"
    assert entries == [
        IndexEntry(id="a", vector=[0.1, 0.2]),
        IndexEntry(id="b", vector=[0.1, 0.2]),
    ]
