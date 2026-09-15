"""The committed dataset must be exactly what the seeded generator produces, so the
data on disk can never silently drift from the code that built it.
"""

import json
from pathlib import Path

from data.generate_questions import build_questions

REPO_ROOT = Path(__file__).parent.parent


def test_committed_questions_match_generator_output() -> None:
    committed_path = REPO_ROOT / "data" / "questions.jsonl"
    committed = [json.loads(line) for line in committed_path.read_text().splitlines() if line]

    generated = build_questions()

    assert committed == generated
