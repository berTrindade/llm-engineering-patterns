from evals_judge.dataset import ReplyCase
from evals_judge.labels import append_label, load_labels, run_label_cli

POLICY = "policy excerpt"


def _case(case_id):
    return ReplyCase(
        id=case_id,
        customer_message=f"message for {case_id}",
        order_number="NW-100001",
        reply=f"reply for {case_id}",
        designed_label="pass",
        flaw_type=None,
    )


def test_append_label_then_load_labels(tmp_path):
    path = tmp_path / "labels.human.jsonl"
    append_label(path, "case-001", "pass", "looks fine")
    append_label(path, "case-002", "fail", "")

    labels = load_labels(path)
    assert labels == {
        "case-001": {"label": "pass", "note": "looks fine"},
        "case-002": {"label": "fail", "note": ""},
    }


def test_append_label_is_append_only(tmp_path):
    path = tmp_path / "labels.human.jsonl"
    append_label(path, "case-001", "pass", "")
    append_label(path, "case-002", "fail", "")
    assert path.read_text().count("\n") == 2


def test_load_labels_missing_file_returns_empty(tmp_path):
    assert load_labels(tmp_path / "does_not_exist.jsonl") == {}


def test_run_label_cli_prompts_for_each_case_and_saves(tmp_path):
    path = tmp_path / "labels.human.jsonl"
    cases = [_case("case-001"), _case("case-002")]
    inputs = iter(["pass", "", "fail", "note text"])

    def fake_input(prompt):
        return next(inputs)

    printed = []
    run_label_cli(cases, POLICY, path, input_func=fake_input, print_func=printed.append)

    labels = load_labels(path)
    assert labels == {
        "case-001": {"label": "pass", "note": ""},
        "case-002": {"label": "fail", "note": "note text"},
    }
    assert any("case-001" in line for line in printed)


def test_run_label_cli_resumes_and_skips_already_labelled(tmp_path):
    path = tmp_path / "labels.human.jsonl"
    append_label(path, "case-001", "pass", "already done")
    cases = [_case("case-001"), _case("case-002")]
    inputs = iter(["fail", "note text"])

    def fake_input(prompt):
        return next(inputs)

    run_label_cli(cases, POLICY, path, input_func=fake_input, print_func=lambda *_: None)

    labels = load_labels(path)
    assert labels["case-001"] == {"label": "pass", "note": "already done"}
    assert labels["case-002"] == {"label": "fail", "note": "note text"}


def test_run_label_cli_reprompts_on_invalid_label(tmp_path):
    path = tmp_path / "labels.human.jsonl"
    cases = [_case("case-001")]
    inputs = iter(["maybe", "pass", ""])

    def fake_input(prompt):
        return next(inputs)

    run_label_cli(cases, POLICY, path, input_func=fake_input, print_func=lambda *_: None)

    labels = load_labels(path)
    assert labels["case-001"]["label"] == "pass"
