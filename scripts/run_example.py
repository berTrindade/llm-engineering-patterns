"""Runs one case end to end with the json_schema approach, against a local model.

Wired to `make run`.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from structured_outputs.extract import extract_json_schema  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    with (REPO_ROOT / "data" / "cases.jsonl").open() as f:
        case = json.loads(f.readline())

    print("--- input email ---")
    print(case["email_text"])
    print()

    result = extract_json_schema(case["case_id"], case["email_text"], model)

    print("--- extracted invoice ---")
    if result.invoice is not None:
        print(result.invoice.model_dump_json(indent=2))
    else:
        print(f"extraction failed: {result.error}")
    print()
    print(
        f"valid first try: {result.valid_first_try}, "
        f"valid after repair: {result.valid_after_repair}, "
        f"latency: {result.latency_ms:.0f}ms, "
        f"tokens: {result.prompt_tokens} prompt / {result.completion_tokens} completion"
    )


if __name__ == "__main__":
    main()
