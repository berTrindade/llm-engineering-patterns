"""`make eval`: run the eval set through retrieval + answering, score it, write results JSON."""

import datetime
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pydantic import BaseModel

from rag_poc.answer import answer_question
from rag_poc.chunk import Chunk
from rag_poc.index import IndexEntry, load_index
from rag_poc.metrics import CaseResult, compute_metrics
from rag_poc.retrieve import embed_query, top_k

QUESTIONS_PATH = ROOT / "evals" / "questions.jsonl"
SNAPSHOT_PATH = ROOT / "data" / "snapshot.json"
INDEX_PATH = ROOT / "index" / "embeddings.json"
RESULTS_DIR = ROOT / "results"


class EvalCase(BaseModel):
    id: str
    question: str
    answerable: bool
    expected_sources: list[str]  # "<source_url> | <heading path>"


def load_cases(path: Path) -> list[EvalCase]:
    lines = path.read_text().strip().split("\n")
    return [EvalCase(**json.loads(line)) for line in lines if line]


def _expected_urls(case: EvalCase) -> set[str]:
    return {source.split(" | ", 1)[0] for source in case.expected_sources}


def run_case(
    case: EvalCase,
    entries: list[IndexEntry],
    chunks_by_id: dict[str, Chunk],
    cheap_model: str,
    embed_model: str,
    threshold: float,
) -> CaseResult:
    try:
        query_vector = embed_query(case.question, embed_model=embed_model)
        retrieved = top_k(query_vector, entries, chunks_by_id, k=5)
        result, llm_response = answer_question(
            question=case.question,
            retrieved=retrieved,
            cheap_model=cheap_model,
            similarity_threshold=threshold,
        )
    except Exception as exc:  # noqa: BLE001 - recorded as a failed case, never swallowed
        return CaseResult(
            id=case.id,
            answerable=case.answerable,
            hit_at_5=False,
            citation_correct_share=None,
            refused=False,
            correct=False,
            latency_ms=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            failed=True,
            failure_reason=str(exc),
        )

    expected_urls = _expected_urls(case)
    hit_at_5 = bool(expected_urls) and any(chunk.source_url in expected_urls for chunk, _score in retrieved)

    citation_correct_share = None
    if case.answerable:
        if result.citations:
            correct = sum(
                1
                for cid in result.citations
                if cid in chunks_by_id and chunks_by_id[cid].source_url in expected_urls
            )
            citation_correct_share = correct / len(result.citations)
        else:
            citation_correct_share = 0.0

    if case.answerable:
        correct = not result.refused
    else:
        correct = result.refused

    return CaseResult(
        id=case.id,
        answerable=case.answerable,
        hit_at_5=hit_at_5,
        citation_correct_share=citation_correct_share,
        refused=result.refused,
        correct=correct,
        latency_ms=llm_response.latency_ms if llm_response else 0.0,
        prompt_tokens=llm_response.prompt_tokens if llm_response else 0,
        completion_tokens=llm_response.completion_tokens if llm_response else 0,
        failed=False,
        failure_reason=None,
    )


def _print_table(metrics: dict) -> None:
    print(f"{'metric':<24} value")
    for key in (
        "hit_rate_at_5",
        "citation_correctness",
        "correct_refusal_rate",
        "false_refusal_rate",
        "mean_latency_ms",
        "mean_tokens",
    ):
        print(f"{key:<24} {metrics[key]}")
    print(f"failing cases: {len(metrics['failing_ids'])}")
    for failure in metrics["failing_ids"]:
        print(f"  {failure['id']}: {failure['reason']}")


def main() -> None:
    cheap_model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    eval_limit = os.environ.get("EVAL_LIMIT")

    cases = load_cases(QUESTIONS_PATH)
    if eval_limit:
        cases = cases[: int(eval_limit)]

    records = json.loads(SNAPSHOT_PATH.read_text())
    chunks_by_id = {r["id"]: Chunk(**{k: v for k, v in r.items() if k != "fetch_date"}) for r in records}
    entries = load_index(INDEX_PATH)

    RESULTS_DIR.mkdir(exist_ok=True)
    today = datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    filename = f"smoke-{today}.json" if eval_limit else f"{today}.json"
    output_path = RESULTS_DIR / filename

    results: list[CaseResult] = []

    def write_results(complete: bool) -> dict:
        metrics = compute_metrics(results)
        output_path.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "cases": [r.model_dump() for r in results],
                    "cheap_model": cheap_model,
                    "embed_model": embed_model,
                    "smoke": bool(eval_limit),
                    "complete": complete,
                },
                indent=2,
            )
        )
        return metrics

    for case in cases:
        results.append(
            run_case(case, entries, chunks_by_id, cheap_model=cheap_model, embed_model=embed_model, threshold=0.3)
        )
        # Written after every case, so a run that stalls or is killed keeps what it finished.
        write_results(complete=False)

    metrics = write_results(complete=True)
    _print_table(metrics)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
