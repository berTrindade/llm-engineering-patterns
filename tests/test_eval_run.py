import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.run import EvalCase, load_cases, run_case
from rag_poc.chunk import Chunk
from rag_poc.index import IndexEntry


def test_load_cases_parses_jsonl(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "Who fixes the boiler?",
                        "answerable": True,
                        "expected_sources": ["https://example.test/page | Page > Repairs"],
                    }
                ),
                json.dumps(
                    {"id": "q2", "question": "Scotland rules?", "answerable": False, "expected_sources": []}
                ),
            ]
        )
    )

    cases = load_cases(path)

    assert len(cases) == 2
    assert isinstance(cases[0], EvalCase)
    assert cases[0].id == "q1"
    assert cases[0].expected_sources == ["https://example.test/page | Page > Repairs"]
    assert cases[1].answerable is False


def test_run_case_records_failure_instead_of_raising(monkeypatch):
    case = EvalCase(id="q1", question="Who fixes the boiler?", answerable=True, expected_sources=[])
    entries = [IndexEntry(id="a", vector=[0.1, 0.2])]
    chunks_by_id = {
        "a": Chunk(id="a", text="text", source_url="https://example.test", page_title="P", heading_path="P > H")
    }

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("evals.run.embed_query", boom)

    result = run_case(
        case=case,
        entries=entries,
        chunks_by_id=chunks_by_id,
        cheap_model="qwen3:8b",
        embed_model="nomic-embed-text",
        threshold=0.3,
    )

    assert result.failed is True
    assert "model exploded" in result.failure_reason
    assert result.id == "q1"
