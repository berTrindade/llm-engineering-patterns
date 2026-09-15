-include .env
export

.PHONY: setup run eval test lint

setup:
	uv sync

run:
	uv run python scripts/run_example.py

eval:
	uv run python evals/run.py

test:
	uv run pytest

lint:
	uv run ruff check .
