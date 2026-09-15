from evals.run import load_tasks


def test_load_tasks_parses_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id": "t1", "category": "simple", "request": "book a room", '
        '"checks": [{"type": "nothing_booked"}]}\n'
        '{"id": "t2", "category": "impossible", "request": "book 50 people", '
        '"checks": [{"type": "nothing_booked"}]}\n'
    )

    tasks = load_tasks(path)

    assert len(tasks) == 2
    assert tasks[0]["id"] == "t1"
    assert tasks[0]["category"] == "simple"
    assert tasks[1]["checks"] == [{"type": "nothing_booked"}]
