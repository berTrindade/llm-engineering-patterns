"""Loads the question set for the eval runners."""

import json
from dataclasses import dataclass


@dataclass
class Question:
    id: str
    question: str
    required_facts: list[str]
    difficulty: str


def load_questions(path: str = "data/questions.jsonl", limit: int | None = None) -> list[Question]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            questions.append(
                Question(
                    id=row["id"],
                    question=row["question"],
                    required_facts=row["required_facts"],
                    difficulty=row["difficulty"],
                )
            )
    if limit is not None:
        questions = questions[:limit]
    return questions
