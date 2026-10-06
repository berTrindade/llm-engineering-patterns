-include .env
export

.PHONY: setup run eval test lint

setup:
	uv sync

run:
	uv run python -m tool_calling_poc.run_example

eval:
	uv run python -m evals.run

test:
	uv run pytest

lint:
	uv run ruff check
