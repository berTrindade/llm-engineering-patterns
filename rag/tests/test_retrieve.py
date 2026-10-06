from rag_poc.chunk import Chunk
from rag_poc.index import IndexEntry
from rag_poc.retrieve import cosine_similarity, top_k


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def _chunk(cid: str) -> Chunk:
    return Chunk(
        id=cid,
        text=f"text for {cid}",
        source_url="https://example.test/page",
        page_title="Page",
        heading_path="Page > Heading",
    )


def test_top_k_returns_closest_sorted_descending():
    entries = [
        IndexEntry(id="a", vector=[1.0, 0.0]),
        IndexEntry(id="b", vector=[0.0, 1.0]),
        IndexEntry(id="c", vector=[0.9, 0.1]),
    ]
    chunks_by_id = {"a": _chunk("a"), "b": _chunk("b"), "c": _chunk("c")}

    results = top_k(query_vector=[1.0, 0.0], entries=entries, chunks_by_id=chunks_by_id, k=2)

    assert [chunk.id for chunk, _score in results] == ["a", "c"]
    assert results[0][1] >= results[1][1]
