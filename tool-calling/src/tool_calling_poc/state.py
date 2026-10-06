from __future__ import annotations

import copy
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel

_SLOT_GRID_MINUTES = 15


def _overlaps(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    return start_a < end_b and start_b < end_a


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

    def list_rooms(self, min_capacity: int = 0, needs: list[str] | None = None) -> list[Room]:
        needs = needs or []
        return [
            room
            for room in self.rooms.values()
            if room.capacity >= min_capacity and all(need in room.equipment for need in needs)
        ]

    def list_meetings(
        self, person: str | None = None, date: date | None = None
    ) -> list[Meeting]:
        result = list(self.meetings.values())
        if person is not None:
            result = [m for m in result if person in m.attendees]
        if date is not None:
            result = [m for m in result if m.start.date() == date]
        return result

    def _attendee_free(self, person_id: str, start: datetime, end: datetime) -> bool:
        for meeting in self.meetings.values():
            if person_id not in meeting.attendees:
                continue
            meeting_end = meeting.start + timedelta(minutes=meeting.duration_minutes)
            if _overlaps(start, end, meeting.start, meeting_end):
                return False
        return True

    def _within_working_hours(self, person: Person, start: datetime, end: datetime) -> bool:
        tz = ZoneInfo(person.timezone)
        local_start = start.astimezone(tz)
        local_end = end.astimezone(tz)
        if local_start.date() != local_end.date():
            return False
        wh_start = time.fromisoformat(person.working_hours[0])
        wh_end = time.fromisoformat(person.working_hours[1])
        return wh_start <= local_start.time() and local_end.time() <= wh_end

    def find_free_slots(
        self,
        attendees: list[str],
        duration_minutes: int,
        earliest: datetime,
        latest: datetime,
    ) -> list[tuple[datetime, datetime]]:
        people = [self.get_person(pid) for pid in attendees]
        duration = timedelta(minutes=duration_minutes)
        step = timedelta(minutes=_SLOT_GRID_MINUTES)
        slots: list[tuple[datetime, datetime]] = []
        start = earliest
        while start + duration <= latest:
            end = start + duration
            if all(self._attendee_free(p.id, start, end) for p in people) and all(
                self._within_working_hours(p, start, end) for p in people
            ):
                slots.append((start, end))
            start += step
        return slots

    def book_meeting(
        self,
        title: str,
        attendees: list[str],
        room_id: str,
        start: datetime,
        duration_minutes: int,
    ) -> Meeting:
        room = self.get_room(room_id)
        people = [self.get_person(pid) for pid in attendees]
        end = start + timedelta(minutes=duration_minutes)

        if len(attendees) > room.capacity:
            raise DomainError(
                f"room {room_id} has capacity {room.capacity}, {len(attendees)} attendees requested"
            )

        for meeting in self.meetings.values():
            if meeting.room_id != room_id:
                continue
            meeting_end = meeting.start + timedelta(minutes=meeting.duration_minutes)
            if _overlaps(start, end, meeting.start, meeting_end):
                raise DomainError(f"room {room_id} is already booked at that time")

        for person in people:
            if not self._attendee_free(person.id, start, end):
                raise DomainError(f"{person.id} already has a meeting at that time")
            if not self._within_working_hours(person, start, end):
                raise DomainError(f"{person.id} is outside working hours at that time")

        meeting = Meeting(
            id=f"m-{uuid4().hex[:8]}",
            title=title,
            attendees=attendees,
            room_id=room_id,
            start=start,
            duration_minutes=duration_minutes,
        )
        self.meetings[meeting.id] = meeting
        return meeting

    def cancel_meeting(self, meeting_id: str) -> None:
        if meeting_id not in self.meetings:
            raise DomainError(f"unknown meeting id: {meeting_id}")
        del self.meetings[meeting_id]
