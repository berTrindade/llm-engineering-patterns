import json
from pathlib import Path

from tool_calling_poc.state import OfficeState
from tool_calling_poc.tools import TOOL_REGISTRY, call_tool, tool_specs_for_llm

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "office.json"


def load_state() -> OfficeState:
    return OfficeState.from_json(DATA_PATH)


def test_tool_specs_for_llm_has_five_tools_with_schemas():
    specs = tool_specs_for_llm()
    assert len(specs) == 5
    names = {spec["function"]["name"] for spec in specs}
    assert names == {
        "find_free_slots",
        "list_rooms",
        "book_meeting",
        "cancel_meeting",
        "list_meetings",
    }
    book_spec = next(s for s in specs if s["function"]["name"] == "book_meeting")
    required = book_spec["function"]["parameters"]["required"]
    assert set(required) == {"title", "attendees", "room_id", "start", "duration_minutes"}


def test_tool_registry_has_five_entries():
    assert set(TOOL_REGISTRY.keys()) == {
        "find_free_slots",
        "list_rooms",
        "book_meeting",
        "cancel_meeting",
        "list_meetings",
    }


def test_call_tool_book_meeting_success():
    state = load_state()
    ok, result = call_tool(
        state,
        "book_meeting",
        {
            "title": "Planning",
            "attendees": ["p01", "p02"],
            "room_id": "room-1",
            "start": "2026-09-15T13:00:00Z",
            "duration_minutes": 30,
        },
    )
    assert ok is True
    body = json.loads(result)
    assert body["title"] == "Planning"
    assert len(state.meetings) == 7  # 6 seeded + 1 new


def test_call_tool_missing_required_field_returns_error():
    state = load_state()
    ok, result = call_tool(
        state,
        "book_meeting",
        {"title": "Missing room", "attendees": ["p01"], "duration_minutes": 30},
    )
    assert ok is False
    assert "room_id" in result


def test_call_tool_domain_error_returns_error():
    state = load_state()
    ok, result = call_tool(
        state,
        "book_meeting",
        {
            "title": "Too many people",
            "attendees": ["p01", "p02", "p03"],
            "room_id": "room-3",
            "start": "2026-09-15T13:00:00Z",
            "duration_minutes": 30,
        },
    )
    assert ok is False
    assert "capacity" in result


def test_call_tool_unknown_tool_name():
    state = load_state()
    ok, result = call_tool(state, "delete_everything", {})
    assert ok is False
    assert "unknown tool" in result


def test_call_tool_list_rooms():
    state = load_state()
    ok, result = call_tool(state, "list_rooms", {"min_capacity": 6})
    assert ok is True
    body = json.loads(result)
    assert {r["id"] for r in body} == {"room-2", "room-4", "room-5"}


def test_call_tool_cancel_meeting():
    state = load_state()
    ok, result = call_tool(state, "cancel_meeting", {"meeting_id": "m01"})
    assert ok is True
    assert "m01" not in state.meetings


def test_call_tool_cancel_unknown_meeting_returns_error():
    state = load_state()
    ok, result = call_tool(state, "cancel_meeting", {"meeting_id": "m-nope"})
    assert ok is False
    assert "unknown meeting id" in result


def test_call_tool_list_meetings():
    state = load_state()
    ok, result = call_tool(state, "list_meetings", {"person": "p01"})
    assert ok is True
    body = json.loads(result)
    assert {m["id"] for m in body} == {"m01", "m03", "m06"}


def test_call_tool_find_free_slots():
    state = load_state()
    ok, result = call_tool(
        state,
        "find_free_slots",
        {
            "attendees": ["p01"],
            "duration_minutes": 30,
            "earliest": "2026-09-15T13:00:00Z",
            "latest": "2026-09-15T14:00:00Z",
        },
    )
    assert ok is True
    body = json.loads(result)
    assert len(body) > 0
    assert "start" in body[0] and "end" in body[0]
