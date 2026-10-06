import json
from pathlib import Path

from evals.run import load_tasks, run_task
from tool_calling_poc.llm import LLMResponse, ToolCall
from tool_calling_poc.state import OfficeState

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "office.json"


def load_state() -> OfficeState:
    return OfficeState.from_json(DATA_PATH)


def scripted_chat_fn(responses: list[LLMResponse]):
    it = iter(responses)

    def chat_fn(**kwargs):
        return next(it)

    return chat_fn


def make_response(content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=5.0,
        model="test-model",
    )


def test_load_tasks_parses_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id": "t1", "category": "simple", "request": "book a room", '
        '"checks": [{"type": "nothing_booked"}]}\n'
        '{"id": "t2", "category": "impossible", "request": "book 50 people", '
        '"checks": [{"type": "nothing_booked"}]}\n'
    )

    tasks = load_tasks(path)

    assert len(tasks) == 2
    assert tasks[0]["id"] == "t1"
    assert tasks[0]["category"] == "simple"
    assert tasks[1]["checks"] == [{"type": "nothing_booked"}]


def test_run_task_records_reason_and_trace_on_wrong_end_state():
    task = {
        "id": "t1",
        "category": "simple",
        "request": "book something",
        "checks": [{"type": "nothing_booked"}],
    }
    book_args = json.dumps(
        {
            "title": "X",
            "attendees": ["p01"],
            "room_id": "room-1",
            "start": "2026-09-16T13:00:00Z",
            "duration_minutes": 30,
        }
    )
    responses = [
        make_response(tool_calls=[ToolCall(id="call_1", name="book_meeting", arguments=book_args)]),
        make_response(content="Booked."),
    ]

    outcome = run_task(task, load_state(), scripted_chat_fn(responses), model="test-model")

    assert outcome.success is False
    assert outcome.failure_type == "wrong_end_state"
    assert outcome.reason is not None
    assert "nothing_booked" in outcome.reason
    assert outcome.final_message == "Booked."
    assert outcome.stopped_reason == "final_message"
    assert len(outcome.trace) == 2
    assert outcome.trace[0]["tool_name"] == "book_meeting"
    assert outcome.trace[0]["is_error"] is False


def test_run_task_records_exception_reason_on_model_error():
    task = {
        "id": "t1",
        "category": "simple",
        "request": "book something",
        "checks": [{"type": "nothing_booked"}],
    }

    def broken_chat_fn(**kwargs):
        raise RuntimeError("boom")

    outcome = run_task(task, load_state(), broken_chat_fn, model="test-model")

    assert outcome.success is False
    assert outcome.failure_type == "model_error"
    assert outcome.reason == "RuntimeError: boom"
    assert outcome.trace == []
