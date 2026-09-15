from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime

from pydantic import BaseModel, ValidationError

from tool_calling_poc.state import DomainError, OfficeState


class FindFreeSlotsArgs(BaseModel):
    attendees: list[str]
    duration_minutes: int
    earliest: datetime
    latest: datetime


class ListRoomsArgs(BaseModel):
    min_capacity: int = 0
    needs: list[str] = []


class BookMeetingArgs(BaseModel):
    title: str
    attendees: list[str]
    room_id: str
    start: datetime
    duration_minutes: int


class CancelMeetingArgs(BaseModel):
    meeting_id: str


class ListMeetingsArgs(BaseModel):
    person: str | None = None
    date: date_type | None = None


def _find_free_slots(state: OfficeState, args: FindFreeSlotsArgs) -> list[dict]:
    slots = state.find_free_slots(args.attendees, args.duration_minutes, args.earliest, args.latest)
    return [{"start": start.isoformat(), "end": end.isoformat()} for start, end in slots]


def _list_rooms(state: OfficeState, args: ListRoomsArgs) -> list[dict]:
    return [room.model_dump() for room in state.list_rooms(args.min_capacity, args.needs)]


def _book_meeting(state: OfficeState, args: BookMeetingArgs) -> dict:
    meeting = state.book_meeting(
        title=args.title,
        attendees=args.attendees,
        room_id=args.room_id,
        start=args.start,
        duration_minutes=args.duration_minutes,
    )
    return meeting.model_dump(mode="json")


def _cancel_meeting(state: OfficeState, args: CancelMeetingArgs) -> dict:
    state.cancel_meeting(args.meeting_id)
    return {"cancelled": args.meeting_id}


def _list_meetings(state: OfficeState, args: ListMeetingsArgs) -> list[dict]:
    meetings = state.list_meetings(person=args.person, date=args.date)
    return [meeting.model_dump(mode="json") for meeting in meetings]


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[OfficeState, BaseModel], object]


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "find_free_slots": ToolSpec(
        name="find_free_slots",
        description=(
            "Find time slots where every given attendee is free and within their working hours."
        ),
        args_model=FindFreeSlotsArgs,
        handler=_find_free_slots,
    ),
    "list_rooms": ToolSpec(
        name="list_rooms",
        description="List meeting rooms, optionally filtered by minimum capacity or equipment.",
        args_model=ListRoomsArgs,
        handler=_list_rooms,
    ),
    "book_meeting": ToolSpec(
        name="book_meeting",
        description="Book a meeting for a given title, attendees, room and start time.",
        args_model=BookMeetingArgs,
        handler=_book_meeting,
    ),
    "cancel_meeting": ToolSpec(
        name="cancel_meeting",
        description="Cancel an existing meeting by id.",
        args_model=CancelMeetingArgs,
        handler=_cancel_meeting,
    ),
    "list_meetings": ToolSpec(
        name="list_meetings",
        description="List meetings, optionally filtered by attendee or date.",
        args_model=ListMeetingsArgs,
        handler=_list_meetings,
    ),
}


def tool_specs_for_llm() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.args_model.model_json_schema(),
            },
        }
        for spec in TOOL_REGISTRY.values()
    ]


def call_tool(state: OfficeState, name: str, arguments: dict) -> tuple[bool, str]:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return False, f"unknown tool: {name}"

    try:
        parsed_args = spec.args_model(**arguments)
    except ValidationError as exc:
        return False, f"invalid arguments: {exc}"

    try:
        result = spec.handler(state, parsed_args)
    except DomainError as exc:
        return False, str(exc)

    return True, json.dumps(result)
