"""CLI for the human-in-the-loop approval workflow: `python -m hitl <command>`.

Every proposal requires an explicit human decision (approve/reject) before anything is
executed. `process` never approves anything itself; it only proposes and stops at
AWAITING_APPROVAL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

from hitl import agent, db

DEFAULT_DB_PATH = "hitl.db"
DEFAULT_POLICY_PATH = "data/policy.md"
DEFAULT_CHEAP_MODEL = "qwen3:8b"


def _db_path() -> str:
    return os.environ.get("HITL_DB_PATH", DEFAULT_DB_PATH)


def _policy_text() -> str:
    path = os.environ.get("HITL_POLICY_PATH", DEFAULT_POLICY_PATH)
    return Path(path).read_text()


def _connect() -> "sqlite3.Connection":  # noqa: F821 - typing only, avoids unused import
    conn = db.get_connection(_db_path())
    db.init_db(conn)
    return conn


def _touch_marker() -> None:
    marker = os.environ.get("HITL_TEST_MARKER", _db_path() + ".marker")
    Path(marker).write_text("ready")


def cmd_submit(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.file).read_text())
    records = data if isinstance(data, list) else [data]
    conn = _connect()
    for record in records:
        request_id = record.get("id") or uuid.uuid4().hex[:8]
        db.insert_request(conn, request_id, record.get("type", "expense"), record)
        print(f"submitted {request_id}")
    conn.close()
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    conn = _connect()
    policy_text = _policy_text()
    model = os.environ.get("CHEAP_MODEL", DEFAULT_CHEAP_MODEL)

    for row in db.list_requests(conn, db.RequestStatus.RECEIVED):
        payload = json.loads(row["payload_json"])
        result = agent.propose(payload, policy_text, model)

        if not result.ok:
            db.transition(
                conn, row["id"], db.RequestStatus.FAILED, actor="agent",
                detail=f"invalid proposal output: {result.error}", fail_reason=result.error,
            )
            print(f"{row['id']}: FAILED ({result.error})")
            continue

        proposal = result.proposal
        db.transition(
            conn, row["id"], db.RequestStatus.PROPOSED, actor="agent",
            detail=f"proposed {proposal.action}",
            proposal_action=proposal.action,
            proposal_amount=proposal.amount,
            proposal_clauses_json=json.dumps(proposal.policy_clauses),
            proposal_rationale=proposal.rationale,
        )
        db.transition(
            conn, row["id"], db.RequestStatus.AWAITING_APPROVAL, actor="agent",
            detail="awaiting human decision",
        )
        print(f"{row['id']}: {proposal.action} -> AWAITING_APPROVAL")

    conn.close()

    if os.environ.get("HITL_TEST_HANG_AFTER_PROCESS") == "1":
        _touch_marker()
        while True:
            time.sleep(3600)

    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    conn = _connect()
    rows = db.list_requests(conn, db.RequestStatus.AWAITING_APPROVAL)
    for row in rows:
        print(
            f"{row['id']}: {row['type']} proposal={row['proposal_action']} "
            f"amount={row['proposal_amount']} rationale={row['proposal_rationale']}"
        )
    conn.close()
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    conn = _connect()
    fields = {"approved_by": args.by}
    if args.amount is not None:
        fields["approved_amount"] = args.amount
    db.transition(
        conn, args.id, db.RequestStatus.APPROVED, actor=f"human:{args.by}",
        detail=f"approved by {args.by}", **fields,
    )
    conn.close()
    print(f"{args.id}: APPROVED")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    conn = _connect()
    db.transition(
        conn, args.id, db.RequestStatus.REJECTED, actor=f"human:{args.by}",
        detail=f"rejected by {args.by}: {args.reason}",
        rejected_by=args.by, reject_reason=args.reason,
    )
    conn.close()
    print(f"{args.id}: REJECTED")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    conn = _connect()
    approved = db.list_requests(conn, db.RequestStatus.APPROVED)
    conn.close()

    if os.environ.get("HITL_TEST_HANG_BEFORE_EXECUTE") == "1":
        _touch_marker()
        while True:
            time.sleep(3600)

    conn = _connect()
    for row in approved:
        outcome = db.execute_request(conn, row["id"])
        print(f"{row['id']}: {outcome}")
    conn.close()
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    conn = _connect()
    for entry in db.get_audit_log(conn, args.id):
        print(f"{entry['ts']} {entry['from_status']} -> {entry['to_status']} "
              f"by {entry['actor']}: {entry['detail']}")
    conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m hitl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="submit a request from a JSON file")
    p_submit.add_argument("file")
    p_submit.set_defaults(func=cmd_submit)

    p_process = sub.add_parser("process", help="run the agent on RECEIVED requests")
    p_process.set_defaults(func=cmd_process)

    p_pending = sub.add_parser("pending", help="list requests awaiting a decision")
    p_pending.set_defaults(func=cmd_pending)

    p_approve = sub.add_parser("approve", help="approve a request")
    p_approve.add_argument("id")
    p_approve.add_argument("--by", required=True)
    p_approve.add_argument("--amount", type=float, default=None)
    p_approve.set_defaults(func=cmd_approve)

    p_reject = sub.add_parser("reject", help="reject a request")
    p_reject.add_argument("id")
    p_reject.add_argument("--by", required=True)
    p_reject.add_argument("--reason", required=True)
    p_reject.set_defaults(func=cmd_reject)

    p_resume = sub.add_parser("resume", help="execute APPROVED requests not yet executed")
    p_resume.set_defaults(func=cmd_resume)

    p_audit = sub.add_parser("audit", help="print the audit log for a request")
    p_audit.add_argument("id")
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
