-include .env
export

.PHONY: setup run eval label test lint

setup:
	uv sync

run:
	uv run python -m evals_judge.dataset
	EVAL_LIMIT=1 uv run python -m evals.run

eval:
	uv run python -m evals.run

label:
	uv run python -m evals_judge.labels

test:
	uv run pytest

lint:
	uv run ruff check .
