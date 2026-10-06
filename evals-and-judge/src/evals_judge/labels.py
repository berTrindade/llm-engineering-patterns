"""Human labelling CLI. Resumable, incremental: each answer is saved before the next prompt."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from evals_judge.dataset import DATA_PATH, ReplyCase, load_dataset

LABELS_PATH = DATA_PATH.parent / "labels.human.jsonl"
POLICY_PATH = DATA_PATH.parent / "policy.md"


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    labels: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                labels[row["case_id"]] = {"label": row["label"], "note": row["note"]}
    return labels


def append_label(path: Path, case_id: str, label: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"case_id": case_id, "label": label, "note": note}) + "\n")


def run_label_cli(
    cases: list[ReplyCase],
    policy_text: str,
    path: Path,
    input_func: Callable[[str], str] = input,
    print_func: Callable[[str], None] = print,
) -> None:
    existing = load_labels(path)
    remaining = [c for c in cases if c.id not in existing]

    if not remaining:
        print_func(f"All {len(cases)} cases already labelled.")
        return

    print_func(f"{len(existing)} of {len(cases)} labelled already, {len(remaining)} to go.")
    for case in remaining:
        print_func(f"\n[{case.id}]")
        print_func(f"Customer message:\n{case.customer_message}")
        print_func(f"\nReply:\n{case.reply}")
        print_func(f"\nPolicy:\n{policy_text}")

        label = ""
        while label not in ("pass", "fail"):
            label = input_func("Pass or fail? [pass/fail] ").strip().lower()
            if label not in ("pass", "fail"):
                print_func("Please answer 'pass' or 'fail'.")

        note = input_func("Note (optional): ").strip()
        append_label(path, case.id, label, note)
        print_func(f"Saved {case.id}: {label}")


def main() -> None:
    cases = load_dataset()
    policy_text = POLICY_PATH.read_text()
    run_label_cli(cases, policy_text, LABELS_PATH)


if __name__ == "__main__":
    main()
