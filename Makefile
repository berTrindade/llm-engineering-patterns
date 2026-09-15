-include .env
export

.PHONY: setup run eval calibrate test lint

setup:
	uv sync

run:
	EVAL_LIMIT=1 RESULTS_FILE=results/run-example.json uv run python -m evals.run

eval:
	uv run python -m evals.run

calibrate:
	uv run python -m context_window_poc.calibrate

test:
	uv run pytest

lint:
	uv run ruff check
