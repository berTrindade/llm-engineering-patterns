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


def test_list_rooms_filters_by_capacity():
    state = load_state()
    rooms = state.list_rooms(min_capacity=6)
    assert {r.id for r in rooms} == {"room-2", "room-4", "room-5"}


def test_list_rooms_filters_by_equipment():
    state = load_state()
    rooms = state.list_rooms(needs=["whiteboard"])
    assert {r.id for r in rooms} == {"room-4", "room-5"}


def test_find_free_slots_excludes_existing_conflict():
    state = load_state()
    # p01 has "Weekly sync" 2026-09-15T10:00Z for 30 minutes
    slots = state.find_free_slots(
        attendees=["p01"],
        duration_minutes=30,
        earliest=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
        latest=datetime(2026, 9, 15, 10, 30, tzinfo=UTC),
    )
    assert slots == []


def test_find_free_slots_excludes_outside_working_hours_cross_timezone():
    state = load_state()
    # 02:00-03:00 UTC is 07:30-08:30 IST, before Dev Patel's 09:30 local start
    slots = state.find_free_slots(
        attendees=["p04"],
        duration_minutes=30,
        earliest=datetime(2026, 9, 16, 2, 0, tzinfo=UTC),
        latest=datetime(2026, 9, 16, 3, 0, tzinfo=UTC),
    )
    assert slots == []


def test_find_free_slots_returns_open_slot():
    state = load_state()
    slots = state.find_free_slots(
        attendees=["p01"],
        duration_minutes=30,
        earliest=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
        latest=datetime(2026, 9, 15, 14, 0, tzinfo=UTC),
    )
    assert len(slots) > 0
    for start, end in slots:
        assert (end - start).total_seconds() == 30 * 60


def test_book_meeting_happy_path():
    state = load_state()
    meeting = state.book_meeting(
        title="Planning",
        attendees=["p01", "p02"],
        room_id="room-1",
        start=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
        duration_minutes=30,
    )
    assert meeting.id in state.meetings
    assert meeting.title == "Planning"


def test_book_meeting_over_capacity_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.book_meeting(
            title="Too many people",
            attendees=["p01", "p02", "p03"],
            room_id="room-3",  # capacity 2
            start=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
            duration_minutes=30,
        )


def test_book_meeting_room_double_booked_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.book_meeting(
            title="Clash",
            attendees=["p06"],
            room_id="room-1",  # m01 already books room-1 at this time
            start=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
            duration_minutes=30,
        )


def test_book_meeting_attendee_double_booked_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.book_meeting(
            title="Clash",
            attendees=["p01"],  # already in m01 at this time
            room_id="room-2",
            start=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
            duration_minutes=30,
        )


def test_book_meeting_outside_working_hours_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.book_meeting(
            title="Too early",
            attendees=["p04"],
            room_id="room-1",
            start=datetime(2026, 9, 16, 2, 0, tzinfo=UTC),
            duration_minutes=30,
        )


def test_book_meeting_unknown_room_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.book_meeting(
            title="Ghost room",
            attendees=["p01"],
            room_id="room-does-not-exist",
            start=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
            duration_minutes=30,
        )


def test_book_meeting_unknown_attendee_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.book_meeting(
            title="Ghost attendee",
            attendees=["p-does-not-exist"],
            room_id="room-1",
            start=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
            duration_minutes=30,
        )


def test_cancel_meeting_removes_it():
    state = load_state()
    state.cancel_meeting("m01")
    assert "m01" not in state.meetings


def test_cancel_meeting_unknown_id_raises():
    state = load_state()
    with pytest.raises(DomainError):
        state.cancel_meeting("m-does-not-exist")


def test_list_meetings_filters_by_person():
    state = load_state()
    meetings = state.list_meetings(person="p01")
    assert {m.id for m in meetings} == {"m01", "m03", "m06"}


def test_list_meetings_filters_by_date():
    state = load_state()
    meetings = state.list_meetings(date=datetime(2026, 9, 15, tzinfo=UTC).date())
    assert {m.id for m in meetings} == {"m01", "m04"}
