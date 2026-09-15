import sqlite3

import pytest

from hitl import db


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    db.init_db(c)
    yield c
    c.close()


def test_insert_request_starts_received(conn):
    db.insert_request(conn, "r1", "expense", {"amount": 10})
    req = db.get_request(conn, "r1")
    assert req["status"] == db.RequestStatus.RECEIVED.value


def test_legal_transition_chain_and_execute(conn):
    db.insert_request(conn, "r1", "expense", {"amount": 10})
    db.transition(conn, "r1", db.RequestStatus.PROPOSED, actor="agent", detail="proposed approve")
    db.transition(
        conn, "r1", db.RequestStatus.AWAITING_APPROVAL, actor="agent", detail="waiting"
    )
    db.transition(
        conn, "r1", db.RequestStatus.APPROVED, actor="human:Priya", detail="approved",
        approved_by="Priya",
    )
    result = db.execute_request(conn, "r1")
    assert result == "executed"
    req = db.get_request(conn, "r1")
    assert req["status"] == db.RequestStatus.EXECUTED.value

    cur = conn.execute("SELECT COUNT(*) FROM ledger WHERE request_id = ?", ("r1",))
    assert cur.fetchone()[0] == 1


def test_illegal_transition_raises(conn):
    db.insert_request(conn, "r1", "expense", {"amount": 10})
    with pytest.raises(db.IllegalTransitionError):
        db.transition(conn, "r1", db.RequestStatus.EXECUTED, actor="agent", detail="skip ahead")


def test_execute_twice_is_idempotent(conn):
    db.insert_request(conn, "r1", "expense", {"amount": 10})
    db.transition(conn, "r1", db.RequestStatus.PROPOSED, actor="agent", detail="proposed")
    db.transition(conn, "r1", db.RequestStatus.AWAITING_APPROVAL, actor="agent", detail="waiting")
    db.transition(
        conn, "r1", db.RequestStatus.APPROVED, actor="human:Priya", detail="approved",
        approved_by="Priya",
    )

    first = db.execute_request(conn, "r1")
    second = db.execute_request(conn, "r1")

    assert first == "executed"
    assert second == "already_executed"
    cur = conn.execute("SELECT COUNT(*) FROM ledger WHERE request_id = ?", ("r1",))
    assert cur.fetchone()[0] == 1


def test_audit_log_ordered_one_row_per_transition(conn):
    db.insert_request(conn, "r1", "expense", {"amount": 10})
    db.transition(conn, "r1", db.RequestStatus.PROPOSED, actor="agent", detail="proposed")
    db.transition(conn, "r1", db.RequestStatus.AWAITING_APPROVAL, actor="agent", detail="waiting")

    log = db.get_audit_log(conn, "r1")
    assert [row["to_status"] for row in log] == [
        db.RequestStatus.RECEIVED.value,
        db.RequestStatus.PROPOSED.value,
        db.RequestStatus.AWAITING_APPROVAL.value,
    ]
    assert log == sorted(log, key=lambda row: row["id"])
