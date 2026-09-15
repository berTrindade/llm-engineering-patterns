from evals_judge.dataset import FLAW_TYPES, ReplyCase, generate_dataset


def test_generate_dataset_is_deterministic():
    a = generate_dataset(seed=42, n=60)
    b = generate_dataset(seed=42, n=60)
    assert [c.__dict__ for c in a] == [c.__dict__ for c in b]


def test_generate_dataset_different_seed_differs():
    a = generate_dataset(seed=42, n=60)
    b = generate_dataset(seed=1, n=60)
    assert [c.reply for c in a] != [c.reply for c in b]


def test_generate_dataset_size_and_unique_ids():
    cases = generate_dataset(seed=42, n=60)
    assert len(cases) == 60
    assert len({c.id for c in cases}) == 60
    assert all(isinstance(c, ReplyCase) for c in cases)


def test_designed_labels_consistent_with_flaw_type():
    cases = generate_dataset(seed=42, n=60)
    for case in cases:
        assert case.designed_label in ("pass", "fail")
        if case.designed_label == "fail":
            assert case.flaw_type in FLAW_TYPES
        else:
            assert case.flaw_type is None


def test_designed_labels_are_balanced():
    cases = generate_dataset(seed=42, n=60)
    n_pass = sum(1 for c in cases if c.designed_label == "pass")
    n_fail = sum(1 for c in cases if c.designed_label == "fail")
    assert n_pass == 30
    assert n_fail == 30
    for flaw in FLAW_TYPES:
        assert sum(1 for c in cases if c.flaw_type == flaw) == 6
