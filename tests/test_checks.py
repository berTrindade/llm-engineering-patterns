from datetime import UTC, datetime
from pathlib import Path

from evals.checks import check_end_state
from tool_calling_poc.state import OfficeState

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "office.json"


def load_state() -> OfficeState:
    return OfficeState.from_json(DATA_PATH)


def test_meeting_exists_passes_when_matching_meeting_present():
    state = load_state()
    ok, reason = check_end_state(
        state,
        set(state.meetings),
        [{"type": "meeting_exists", "attendees": ["p01", "p02"], "room_id": "room-1"}],
    )
    assert ok is True
    assert reason is None


def test_meeting_exists_fails_when_no_matching_meeting():
    state = load_state()
    ok, reason = check_end_state(
        state,
        set(state.meetings),
        [{"type": "meeting_exists", "title_contains": "Nonexistent Title Xyz"}],
    )
    assert ok is False
    assert reason is not None


def test_nothing_booked_passes_with_no_new_meetings():
    state = load_state()
    ok, reason = check_end_state(state, set(state.meetings), [{"type": "nothing_booked"}])
    assert ok is True


def test_nothing_booked_fails_after_a_booking():
    state = load_state()
    original_ids = set(state.meetings)
    state.book_meeting(
        title="Extra",
        attendees=["p01"],
        room_id="room-1",
        start=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
        duration_minutes=30,
    )
    ok, reason = check_end_state(state, original_ids, [{"type": "nothing_booked"}])
    assert ok is False
    assert reason is not None


def test_meeting_cancelled_passes_once_removed():
    state = load_state()
    state.cancel_meeting("m01")
    ok, reason = check_end_state(
        state, set(state.meetings), [{"type": "meeting_cancelled", "meeting_id": "m01"}]
    )
    assert ok is True


def test_meeting_cancelled_fails_when_still_present():
    state = load_state()
    ok, reason = check_end_state(
        state, set(state.meetings), [{"type": "meeting_cancelled", "meeting_id": "m01"}]
    )
    assert ok is False
    assert reason is not None
