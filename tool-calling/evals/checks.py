from __future__ import annotations

from tool_calling_poc.state import Meeting, OfficeState


def _meeting_matches(meeting: Meeting, check: dict) -> bool:
    attendees = set(check.get("attendees", []))
    if attendees and not attendees.issubset(set(meeting.attendees)):
        return False
    room_id = check.get("room_id")
    if room_id and meeting.room_id != room_id:
        return False
    title_contains = check.get("title_contains")
    if title_contains and title_contains.lower() not in meeting.title.lower():
        return False
    duration_minutes = check.get("duration_minutes")
    if duration_minutes and meeting.duration_minutes != duration_minutes:
        return False
    return True


def check_end_state(
    state: OfficeState, original_meeting_ids: set[str], checks: list[dict]
) -> tuple[bool, str | None]:
    for check in checks:
        check_type = check["type"]

        if check_type == "meeting_exists":
            if not any(_meeting_matches(m, check) for m in state.meetings.values()):
                return False, f"meeting_exists check failed, no meeting matched: {check}"

        elif check_type == "nothing_booked":
            new_ids = set(state.meetings) - original_meeting_ids
            if new_ids:
                return False, f"nothing_booked check failed, new meetings were booked: {new_ids}"

        elif check_type == "meeting_cancelled":
            meeting_id = check["meeting_id"]
            if meeting_id in state.meetings:
                return False, f"meeting_cancelled check failed, {meeting_id} still exists"

        else:
            return False, f"unknown check type: {check_type}"

    return True, None
