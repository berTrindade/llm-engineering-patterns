import json
from pathlib import Path

from tool_calling_poc.agent import run_agent_loop
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


def test_happy_path_one_tool_call_then_final_message():
    state = load_state()
    responses = [
        make_response(
            tool_calls=[
                ToolCall(id="call_1", name="list_rooms", arguments='{"min_capacity": 4}')
            ]
        ),
        make_response(content="Room Puffin seats 8."),
    ]

    result = run_agent_loop(
        state,
        "which room fits 4 people?",
        scripted_chat_fn(responses),
        model="test-model",
    )

    assert result.final_message == "Room Puffin seats 8."
    assert result.tool_call_count == 1
    assert result.iterations == 2
    assert result.stopped_reason == "final_message"
    assert result.recovered_error_count == 0
    assert result.trace[0].tool_name == "list_rooms"
    assert result.trace[0].is_error is False


def test_recovers_from_invalid_tool_arguments():
    state = load_state()
    bad_args = json.dumps(
        {
            "title": "Sync",
            "attendees": ["p01"],
            "room_id": "room-does-not-exist",
            "start": "2026-09-15T13:00:00Z",
            "duration_minutes": 30,
        }
    )
    good_args = json.dumps(
        {
            "title": "Sync",
            "attendees": ["p01"],
            "room_id": "room-1",
            "start": "2026-09-15T13:00:00Z",
            "duration_minutes": 30,
        }
    )
    responses = [
        make_response(tool_calls=[ToolCall(id="call_1", name="book_meeting", arguments=bad_args)]),
        make_response(tool_calls=[ToolCall(id="call_2", name="book_meeting", arguments=good_args)]),
        make_response(content="Booked Sync for Alice."),
    ]

    result = run_agent_loop(
        state,
        "book a sync for Alice at 13:00",
        scripted_chat_fn(responses),
        model="test-model",
    )

    assert result.recovered_error_count == 1
    assert result.tool_call_count == 2
    assert result.stopped_reason == "final_message"
    assert any(m.title == "Sync" for m in state.meetings.values())
    assert result.trace[0].is_error is True
    assert result.trace[1].is_error is False


def test_iteration_cap_when_model_never_finishes():
    state = load_state()

    def chat_fn(**kwargs):
        return make_response(
            tool_calls=[ToolCall(id="call_x", name="list_rooms", arguments="{}")]
        )

    result = run_agent_loop(
        state,
        "keep going forever",
        chat_fn,
        model="test-model",
        max_iterations=3,
    )

    assert result.stopped_reason == "iteration_cap"
    assert result.iterations == 3
    assert result.final_message is None
    assert result.tool_call_count == 3
