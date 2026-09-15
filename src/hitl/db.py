"""SQLite-backed durable state machine for expense/refund requests.

Every state transition is one SQLite transaction and appends one audit_log row, so the
history of a request can always be replayed from the database alone.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum


class RequestStatus(StrEnum):
    RECEIVED = "RECEIVED"
    PROPOSED = "PROPOSED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class IllegalTransitionError(Exception):
    pass


ALLOWED_TRANSITIONS: dict[RequestStatus, set[RequestStatus]] = {
    RequestStatus.RECEIVED: {RequestStatus.PROPOSED, RequestStatus.FAILED},
    RequestStatus.PROPOSED: {RequestStatus.AWAITING_APPROVAL, RequestStatus.FAILED},
    RequestStatus.AWAITING_APPROVAL: {RequestStatus.APPROVED, RequestStatus.REJECTED},
    RequestStatus.APPROVED: {RequestStatus.EXECUTED, RequestStatus.FAILED},
    RequestStatus.REJECTED: set(),
    RequestStatus.EXECUTED: set(),
    RequestStatus.FAILED: set(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    proposal_action TEXT,
    proposal_amount REAL,
    proposal_clauses_json TEXT,
    proposal_rationale TEXT,
    approved_by TEXT,
    approved_amount REAL,
    rejected_by TEXT,
    reject_reason TEXT,
    fail_reason TEXT
);

CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE REFERENCES requests(id),
    amount REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    from_status TEXT,
    to_status TEXT NOT NULL,
    actor TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def insert_request(conn: sqlite3.Connection, request_id: str, type_: str, payload: dict) -> None:
    now = _now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT INTO requests (id, type, payload_json, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (request_id, type_, json.dumps(payload), RequestStatus.RECEIVED.value, now, now),
        )
        conn.execute(
            """INSERT INTO audit_log (ts, request_id, from_status, to_status, actor, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now, request_id, None, RequestStatus.RECEIVED.value, "system", "request submitted"),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def get_request(conn: sqlite3.Connection, request_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    return dict(row) if row else None


def list_requests(conn: sqlite3.Connection, status: RequestStatus | None = None) -> list[dict]:
    if status is None:
        rows = conn.execute("SELECT * FROM requests ORDER BY created_at").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM requests WHERE status = ? ORDER BY created_at", (status.value,)
        ).fetchall()
    return [dict(row) for row in rows]


def transition(
    conn: sqlite3.Connection,
    request_id: str,
    to_status: RequestStatus,
    actor: str,
    detail: str,
    **fields,
) -> None:
    """Move a request to `to_status` in one transaction, appending one audit_log row.

    Raises IllegalTransitionError if the move is not allowed from the request's current status.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT status FROM requests WHERE id = ?", (request_id,)).fetchone()
        if row is None:
            raise IllegalTransitionError(f"no such request: {request_id}")
        current = RequestStatus(row["status"])
        if to_status not in ALLOWED_TRANSITIONS[current]:
            raise IllegalTransitionError(f"cannot move {request_id} from {current} to {to_status}")

        set_clauses = ["status = ?", "updated_at = ?"]
        params: list = [to_status.value, _now()]
        for key, value in fields.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        params.append(request_id)
        conn.execute(f"UPDATE requests SET {', '.join(set_clauses)} WHERE id = ?", params)

        conn.execute(
            """INSERT INTO audit_log (ts, request_id, from_status, to_status, actor, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (_now(), request_id, current.value, to_status.value, actor, detail),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def execute_request(conn: sqlite3.Connection, request_id: str) -> str:
    """Apply the approved request to the ledger and mark it EXECUTED, exactly once.

    Returns "executed" the first time, "already_executed" on any later call for the same
    request id (guarded by the UNIQUE constraint on ledger.request_id, not by a pre-check,
    so this is safe even if two callers race).
    """
    row = conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
    if row is None:
        raise IllegalTransitionError(f"no such request: {request_id}")
    current = RequestStatus(row["status"])
    if current == RequestStatus.EXECUTED:
        return "already_executed"
    if current != RequestStatus.APPROVED:
        raise IllegalTransitionError(f"cannot execute {request_id} from {current}")

    amount = (
        row["approved_amount"] if row["approved_amount"] is not None else row["proposal_amount"]
    )
    now = _now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO ledger (request_id, amount, created_at) VALUES (?, ?, ?)",
            (request_id, amount or 0.0, now),
        )
        conn.execute(
            "UPDATE requests SET status = ?, updated_at = ? WHERE id = ?",
            (RequestStatus.EXECUTED.value, now, request_id),
        )
        conn.execute(
            """INSERT INTO audit_log (ts, request_id, from_status, to_status, actor, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                now, request_id, current.value, RequestStatus.EXECUTED.value,
                "system", "ledger entry applied",
            ),
        )
        conn.commit()
        return "executed"
    except sqlite3.IntegrityError:
        conn.rollback()
        return "already_executed"
    except BaseException:
        conn.rollback()
        raise


def get_audit_log(conn: sqlite3.Connection, request_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM audit_log WHERE request_id = ? ORDER BY id ASC", (request_id,)
    ).fetchall()
    return [dict(row) for row in rows]
