-include .env
export

.PHONY: setup run eval test lint

setup:
	uv sync

run:
	uv run python -m hitl submit data/sample_requests/expense_travel_ok.json
	uv run python -m hitl submit data/sample_requests/expense_no_receipt.json
	uv run python -m hitl submit data/sample_requests/refund_within_window.json
	uv run python -m hitl submit data/sample_requests/refund_expired_window.json
	uv run python -m hitl process
	uv run python -m hitl pending

eval:
	uv run python -m evals.run

test:
	uv run pytest

lint:
	uv run ruff check
