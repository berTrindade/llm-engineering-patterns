from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


class DomainError(Exception):
    """A scheduling rule was violated (unknown id, conflict, capacity, working hours)."""


class Room(BaseModel):
    id: str
    name: str
    capacity: int
    equipment: list[str] = []


class Person(BaseModel):
    id: str
    name: str
    timezone: str
    working_hours: tuple[str, str]


class Meeting(BaseModel):
    id: str
    title: str
    attendees: list[str]
    room_id: str
    start: datetime
    duration_minutes: int


class OfficeState:
    def __init__(
        self,
        rooms: dict[str, Room],
        people: dict[str, Person],
        meetings: dict[str, Meeting],
        now: datetime,
    ) -> None:
        self.rooms = rooms
        self.people = people
        self.meetings = meetings
        self.now = now

    @classmethod
    def from_json(cls, path: str | Path) -> OfficeState:
        raw = json.loads(Path(path).read_text())
        rooms = {r["id"]: Room(**r) for r in raw["rooms"]}
        people = {p["id"]: Person(**p) for p in raw["people"]}
        meetings = {m["id"]: Meeting(**m) for m in raw["meetings"]}
        now = datetime.fromisoformat(raw["now"].replace("Z", "+00:00"))
        return cls(rooms=rooms, people=people, meetings=meetings, now=now)

    def clone(self) -> OfficeState:
        return copy.deepcopy(self)

    def get_room(self, room_id: str) -> Room:
        try:
            return self.rooms[room_id]
        except KeyError:
            raise DomainError(f"unknown room id: {room_id}") from None

    def get_person(self, person_id: str) -> Person:
        try:
            return self.people[person_id]
        except KeyError:
            raise DomainError(f"unknown person id: {person_id}") from None
