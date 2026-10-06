"""The headline test: kill the process mid-approval and again mid-execution, restart, resume,
and prove nothing is lost, nothing double-executes, and the audit log is complete and ordered.

Uses a recorded proposal fixture (LLM_FIXTURE) so it needs no model and is deterministic. The
two SIGKILL points are made deterministic with a marker file the child process writes right
before it hangs, instead of racing against a sleep.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from hitl import db
from hitl.__main__ import main

FIXTURES = Path(__file__).parent / "fixtures"
POLICY_PATH = Path(__file__).parent.parent / "data" / "policy.md"
REPO_ROOT = Path(__file__).parent.parent

REQUEST_IDS = [f"crash-{i}" for i in range(5)]
APPROVE_IDS = REQUEST_IDS[:3]
REJECT_IDS = REQUEST_IDS[3:]


def _poll_marker(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"marker file never appeared: {path}")


def _sigkill_and_wait(proc: subprocess.Popen, timeout: float = 5.0) -> None:
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=timeout)


def _submit_requests(tmp_path: Path, env: dict) -> None:
    for i, request_id in enumerate(REQUEST_IDS):
        payload = {
            "id": request_id,
            "type": "expense",
            "category": "meals",
            "amount": 30.0,
            "receipt_provided": True,
            "description": f"crash test request {i}",
        }
        req_file = tmp_path / f"{request_id}.json"
        req_file.write_text(json.dumps(payload))
        rc = main(["submit", str(req_file)])
        assert rc == 0


def _run_subprocess_and_kill(env: dict, args: list[str], marker: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, "-m", "hitl", *args],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _poll_marker(marker, timeout=15.0)
    finally:
        _sigkill_and_wait(proc, timeout=10.0)


def test_crash_between_process_and_approval_and_between_approval_and_execution(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "hitl.db"
    marker_process = tmp_path / "process.marker"
    marker_resume = tmp_path / "resume.marker"

    base_env = {
        **os.environ,
        "HITL_DB_PATH": str(db_path),
        "HITL_POLICY_PATH": str(POLICY_PATH),
    }

    monkeypatch.setenv("HITL_DB_PATH", str(db_path))
    monkeypatch.setenv("HITL_POLICY_PATH", str(POLICY_PATH))
    _submit_requests(tmp_path, base_env)

    # Kill #1: mid-batch, right after every request reaches AWAITING_APPROVAL.
    process_env = {
        **base_env,
        "LLM_FIXTURE": str(FIXTURES / "proposal_approve_batch.json"),
        "HITL_TEST_HANG_AFTER_PROCESS": "1",
        "HITL_TEST_MARKER": str(marker_process),
    }
    _run_subprocess_and_kill(process_env, ["process"], marker_process)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for request_id in REQUEST_IDS:
        row = db.get_request(conn, request_id)
        assert row["status"] == db.RequestStatus.AWAITING_APPROVAL.value
        log = db.get_audit_log(conn, request_id)
        assert [entry["to_status"] for entry in log] == [
            db.RequestStatus.RECEIVED.value,
            db.RequestStatus.PROPOSED.value,
            db.RequestStatus.AWAITING_APPROVAL.value,
        ]
    conn.close()

    # Human decisions: approve 3, reject 2, through the same CLI a person uses, against the
    # same database the killed subprocess left behind.
    for request_id in APPROVE_IDS:
        rc = main(["approve", request_id, "--by", "Priya"])
        assert rc == 0
    for request_id in REJECT_IDS:
        rc = main(["reject", request_id, "--by", "Priya", "--reason", "over budget"])
        assert rc == 0

    # Kill #2: immediately after approval, before execution.
    resume_env = {
        **base_env,
        "HITL_TEST_HANG_BEFORE_EXECUTE": "1",
        "HITL_TEST_MARKER": str(marker_resume),
    }
    _run_subprocess_and_kill(resume_env, ["resume"], marker_resume)

    conn = sqlite3.connect(db_path)
    ledger_count = conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    conn.close()
    assert ledger_count == 0, "nothing should have executed before the resume hook hung"

    # Restart: run resume for real, twice, to prove idempotence.
    rc = main(["resume"])
    assert rc == 0
    rc = main(["resume"])
    assert rc == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    ledger_rows = conn.execute("SELECT request_id FROM ledger").fetchall()
    assert sorted(r["request_id"] for r in ledger_rows) == sorted(APPROVE_IDS)
    assert len(ledger_rows) == len(APPROVE_IDS), "no request executed more than once"

    for request_id in APPROVE_IDS:
        row = db.get_request(conn, request_id)
        assert row["status"] == db.RequestStatus.EXECUTED.value
    for request_id in REJECT_IDS:
        row = db.get_request(conn, request_id)
        assert row["status"] == db.RequestStatus.REJECTED.value

    for request_id in REQUEST_IDS:
        log = db.get_audit_log(conn, request_id)
        assert len(log) > 0
        assert log == sorted(log, key=lambda entry: entry["id"])
        for previous, current in zip(log, log[1:]):
            assert previous["to_status"] == current["from_status"], (
                f"audit gap for {request_id}: {previous['to_status']} -> {current['from_status']}"
            )

    # The human decision itself, not just the eventual outcome, must name who made it.
    for request_id in APPROVE_IDS:
        log = db.get_audit_log(conn, request_id)
        approved_entry = next(e for e in log if e["to_status"] == db.RequestStatus.APPROVED.value)
        assert approved_entry["actor"] == "human:Priya"
    for request_id in REJECT_IDS:
        log = db.get_audit_log(conn, request_id)
        rejected_entry = next(e for e in log if e["to_status"] == db.RequestStatus.REJECTED.value)
        assert rejected_entry["actor"] == "human:Priya"

    conn.close()
