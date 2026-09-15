import json
import sqlite3
from pathlib import Path

import pytest

from hitl import db
from hitl.__main__ import main

FIXTURES = Path(__file__).parent / "fixtures"
POLICY_PATH = Path(__file__).parent.parent / "data" / "policy.md"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "hitl.db"
    monkeypatch.setenv("HITL_DB_PATH", str(path))
    monkeypatch.setenv("HITL_POLICY_PATH", str(POLICY_PATH))
    return path


def _request_file(tmp_path, request_id="r1", **overrides):
    payload = {
        "id": request_id,
        "type": "expense",
        "category": "travel",
        "amount": 42.5,
        "receipt_provided": True,
        "description": "train ticket to client site",
    }
    payload.update(overrides)
    path = tmp_path / f"{request_id}.json"
    path.write_text(json.dumps(payload))
    return path


def _status(db_path, request_id):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = db.get_request(conn, request_id)
    conn.close()
    return row["status"] if row else None


def test_submit_inserts_received(db_path, tmp_path):
    req_file = _request_file(tmp_path, "r1")
    rc = main(["submit", str(req_file)])
    assert rc == 0
    assert _status(db_path, "r1") == db.RequestStatus.RECEIVED.value


def test_process_pending_approve_resume(db_path, tmp_path, monkeypatch, capsys):
    main(["submit", str(_request_file(tmp_path, "r1"))])

    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_approve.json"))
    rc = main(["process"])
    assert rc == 0
    assert _status(db_path, "r1") == db.RequestStatus.AWAITING_APPROVAL.value

    capsys.readouterr()
    main(["pending"])
    out = capsys.readouterr().out
    assert "r1" in out
    assert "approve" in out

    rc = main(["approve", "r1", "--by", "Priya"])
    assert rc == 0
    assert _status(db_path, "r1") == db.RequestStatus.APPROVED.value

    rc = main(["resume"])
    assert rc == 0
    assert _status(db_path, "r1") == db.RequestStatus.EXECUTED.value

    conn = sqlite3.connect(db_path)
    ledger_count = conn.execute("SELECT COUNT(*) FROM ledger WHERE request_id = 'r1'").fetchone()[0]
    conn.close()
    assert ledger_count == 1

    rc = main(["resume"])
    assert rc == 0
    conn = sqlite3.connect(db_path)
    ledger_count = conn.execute("SELECT COUNT(*) FROM ledger WHERE request_id = 'r1'").fetchone()[0]
    conn.close()
    assert ledger_count == 1


def test_reject(db_path, tmp_path, monkeypatch):
    main(["submit", str(_request_file(tmp_path, "r2"))])
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_approve.json"))
    main(["process"])

    rc = main(["reject", "r2", "--by", "Priya", "--reason", "no receipt"])
    assert rc == 0
    assert _status(db_path, "r2") == db.RequestStatus.REJECTED.value


def test_audit_prints_ordered_transitions(db_path, tmp_path, monkeypatch, capsys):
    main(["submit", str(_request_file(tmp_path, "r3"))])
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_approve.json"))
    main(["process"])

    capsys.readouterr()
    main(["audit", "r3"])
    out = capsys.readouterr().out
    assert "RECEIVED" in out
    assert "PROPOSED" in out
    assert "AWAITING_APPROVAL" in out
    assert out.index("RECEIVED") < out.index("PROPOSED") < out.index("AWAITING_APPROVAL")


def test_process_invalid_output_fails(db_path, tmp_path, monkeypatch):
    main(["submit", str(_request_file(tmp_path, "r4"))])
    monkeypatch.setenv("LLM_FIXTURE", str(FIXTURES / "proposal_invalid.json"))
    rc = main(["process"])
    assert rc == 0
    assert _status(db_path, "r4") == db.RequestStatus.FAILED.value
