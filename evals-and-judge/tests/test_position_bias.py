from evals_judge.dataset import FLAW_TYPES
from evals_judge.judge import PairVerdict
from evals_judge.position_bias import Pair, generate_pairs, run_position_bias

POLICY = "policy text"


def test_generate_pairs_deterministic_and_balanced():
    a = generate_pairs(seed=7, n=20)
    b = generate_pairs(seed=7, n=20)
    assert [p.__dict__ for p in a] == [p.__dict__ for p in b]
    assert len(a) == 20
    for pair in a:
        assert isinstance(pair, Pair)
        assert pair.reply_better != pair.reply_worse
        assert pair.flaw_type in FLAW_TYPES
    for flaw in FLAW_TYPES:
        assert sum(1 for p in a if p.flaw_type == flaw) == 4


def test_generate_pairs_different_seed_differs():
    a = generate_pairs(seed=7, n=20)
    b = generate_pairs(seed=1, n=20)
    assert [p.reply_better for p in a] != [p.reply_better for p in b]


def _always_a_judge(pair_id, message, reply_a, reply_b, policy_text, model):
    return PairVerdict(pair_id=pair_id, winner="A", model=model, latency_ms=1.0)


def test_run_position_bias_flips_when_judge_always_picks_position_a():
    pairs = generate_pairs(seed=7, n=5)
    result = run_position_bias(pairs, POLICY, "fake-model", judge_pair_fn=_always_a_judge)
    assert result["flip_rate"] == 1.0
    assert result["first_position_win_share"] == 1.0
    assert len(result["records"]) == 5


def test_run_position_bias_no_flip_when_judge_follows_content():
    pairs = generate_pairs(seed=7, n=5)

    def content_judge(pair_id, message, reply_a, reply_b, policy_text, model):
        pair = next(p for p in pairs if pair_id.startswith(p.id))
        winner = "A" if reply_a == pair.reply_better else "B"
        return PairVerdict(pair_id=pair_id, winner=winner, model=model, latency_ms=1.0)

    result = run_position_bias(pairs, POLICY, "fake-model", judge_pair_fn=content_judge)
    assert result["flip_rate"] == 0.0


def test_run_position_bias_skips_errored_runs_in_flip_rate():
    pairs = generate_pairs(seed=7, n=2)

    def erroring_judge(pair_id, message, reply_a, reply_b, policy_text, model):
        return PairVerdict(pair_id=pair_id, winner=None, model=model, error="boom")

    result = run_position_bias(pairs, POLICY, "fake-model", judge_pair_fn=erroring_judge)
    assert result["flip_rate"] == 0.0
    assert result["first_position_win_share"] == 0.0
    assert all(r["flipped"] is None for r in result["records"])
