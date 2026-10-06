import json
from pathlib import Path

from evals.cases import load_questions


def test_load_questions_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    rows = [
        {"id": "q1", "question": "A?", "required_facts": ["a"], "difficulty": "simple_lookup"},
        {"id": "q2", "question": "B?", "required_facts": ["b"], "difficulty": "policy_exception"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    questions = load_questions(str(path))

    assert len(questions) == 2
    assert questions[0].id == "q1"
    assert questions[0].required_facts == ["a"]
    assert questions[1].difficulty == "policy_exception"


def test_load_questions_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    rows = [
        {"id": f"q{i}", "question": "?", "required_facts": [], "difficulty": "simple_lookup"}
        for i in range(5)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    questions = load_questions(str(path), limit=2)

    assert len(questions) == 2
