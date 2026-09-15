"""Answer a question from retrieved chunks only, with citations, or refuse."""

import json

from pydantic import BaseModel

from rag_poc.chunk import Chunk
from rag_poc.llm import LLMResponse, chat_completion

REFUSAL_ANSWER = "I cannot answer this from the private renting guidance I have."

ANSWER_JSON_SCHEMA = {
    "name": "renting_answer",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array", "items": {"type": "string"}},
            "refused": {"type": "boolean"},
        },
        "required": ["answer", "citations", "refused"],
    },
}

SYSTEM_PROMPT = (
    "You answer questions about private renting in England using only the numbered guidance "
    "chunks provided. Cite the id of every chunk you rely on in `citations`. If the chunks do "
    "not cover the question, set `refused` to true and use the exact refusal text you are given."
)


class AnswerResult(BaseModel):
    answer: str
    citations: list[str]
    refused: bool


def _build_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    chunk_lines = "\n\n".join(
        f"[{i + 1}] id={chunk.id}\nheading: {chunk.heading_path}\n{chunk.text}"
        for i, (chunk, _score) in enumerate(retrieved)
    )
    return (
        f"Guidance chunks:\n\n{chunk_lines}\n\n"
        f"Question: {question}\n\n"
        f'If the chunks do not answer the question, respond with answer="{REFUSAL_ANSWER}", '
        "citations=[], refused=true."
    )


def answer_question(
    question: str,
    retrieved: list[tuple[Chunk, float]],
    cheap_model: str,
    similarity_threshold: float = 0.3,
) -> tuple[AnswerResult, LLMResponse | None]:
    best_similarity = max((score for _chunk, score in retrieved), default=0.0)
    if best_similarity < similarity_threshold:
        return AnswerResult(answer=REFUSAL_ANSWER, citations=[], refused=True), None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(question, retrieved)},
    ]
    response = chat_completion(
        messages=messages,
        model=cheap_model,
        response_format={"type": "json_schema", "json_schema": ANSWER_JSON_SCHEMA},
    )
    result = AnswerResult(**json.loads(response.content))
    return result, response
