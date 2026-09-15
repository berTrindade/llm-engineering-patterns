from datetime import UTC, datetime
from pathlib import Path

import pytest

from tool_calling_poc.state import DomainError, OfficeState

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "office.json"


def load_state() -> OfficeState:
    return OfficeState.from_json(DATA_PATH)


def test_from_json_loads_seed():
    state = load_state()
    assert len(state.rooms) == 5
    assert len(state.people) == 12
    assert state.now == datetime(2026, 9, 15, 9, 0, tzinfo=UTC)


def test_clone_is_independent():
    state = load_state()
    clone = state.clone()
    del clone.meetings["m01"]
    assert "m01" in state.meetings
    assert "m01" not in clone.meetings


def test_get_room_unknown_id_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.get_room("room-does-not-exist")


def test_get_person_unknown_id_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.get_person("p-does-not-exist")


def test_get_room_known_id():
    state = load_state()
    room = state.get_room("room-1")
    assert room.name == "Kittiwake"


def test_get_person_known_id():
    state = load_state()
    person = state.get_person("p01")
    assert person.name == "Alice Byrne"
