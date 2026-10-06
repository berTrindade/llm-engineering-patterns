-include .env
export

.PHONY: setup run eval test lint data index

setup:
	uv sync

data:
	uv run python scripts/fetch_data.py

index:
	uv run python scripts/build_index.py

index/embeddings.json:
	uv run python scripts/build_index.py

run: index/embeddings.json
	uv run python -m rag_poc.cli

eval: index/embeddings.json
	uv run python evals/run.py

test:
	uv run pytest

lint:
	uv run ruff check .
