"""Fits the estimator's (a, b, c) coefficients against real API usage.

Model: tokens ~= a*chars + b*n_messages + c, fitted by ordinary least squares on
(chars, n_messages, real_tokens) samples. Solved with plain Gaussian elimination on the
3x3 normal-equations system, no numpy needed for three unknowns.

Run via `make calibrate`. Only fit_tokens_model and held_out_error are covered by
tests; main() makes real calls to CHEAP_MODEL against real thread data.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from context_window_poc.llm import chat_completion
from context_window_poc.tokens import CalibrationFit

# Large prompts take longer to prefill than the default timeout allows for.
CALIBRATION_CALL_TIMEOUT_S = 300.0
# We only need real usage, not a full answer, so cap completion length to keep calls fast.
CALIBRATION_MAX_TOKENS = 8


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gauss-Jordan elimination with partial pivoting for a 3x3 system."""
    n = 3
    augmented = [row[:] + [vector[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("samples do not vary enough in chars/n_messages to fit a, b, c")
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col] / pivot
            for k in range(col, n + 1):
                augmented[row][k] -= factor * augmented[col][k]

    return [augmented[i][n] / augmented[i][i] for i in range(n)]


def fit_tokens_model(samples: list[tuple[int, int, int]]) -> CalibrationFit:
    """samples = [(chars, n_messages, real_tokens), ...]. Least squares fit of
    tokens ~= a*chars + b*n_messages + c via the normal equations."""
    n = len(samples)
    sum_x = sum(chars for chars, _, _ in samples)
    sum_x2 = sum(chars * chars for chars, _, _ in samples)
    sum_m = sum(msgs for _, msgs, _ in samples)
    sum_m2 = sum(msgs * msgs for _, msgs, _ in samples)
    sum_xm = sum(chars * msgs for chars, msgs, _ in samples)
    sum_y = sum(tok for _, _, tok in samples)
    sum_xy = sum(chars * tok for chars, _, tok in samples)
    sum_my = sum(msgs * tok for _, msgs, tok in samples)

    matrix = [
        [sum_x2, sum_xm, sum_x],
        [sum_xm, sum_m2, sum_m],
        [sum_x, sum_m, n],
    ]
    vector = [sum_xy, sum_my, sum_y]
    a, b, c = _solve_3x3(matrix, vector)
    return CalibrationFit(a=a, b=b, c=c)


def held_out_error(fit: CalibrationFit, samples: list[tuple[int, int, int]]) -> float:
    """Mean absolute percentage error of fit against held-out (chars, n_messages,
    real_tokens) samples."""
    errors = []
    for chars, n_messages, real in samples:
        if real == 0:
            continue
        estimate = max(1, round(fit.a * chars + fit.b * n_messages + fit.c))
        errors.append(abs(estimate - real) / real * 100)
    return statistics.fmean(errors) if errors else 0.0


def _build_sample_messages(system_prompt: str, turns: list[dict], target_chars: int) -> list[dict]:
    """Takes a prefix of turns whose cumulative content is close to target_chars, so
    samples span a range of both character counts and message counts."""
    messages = [{"role": "system", "content": system_prompt}]
    total = len(system_prompt)
    for turn in turns:
        if total >= target_chars:
            break
        messages.append(turn)
        total += len(turn["content"])
    return messages


def _collect_samples(model: str, threads: list[dict], target_sizes: list[int]) -> list[tuple[int, int, int]]:
    system_prompt = (
        "You are a Ledgerline support agent. Reply with a single word acknowledging "
        "you have read the conversation."
    )
    samples = []
    for i, target_chars in enumerate(target_sizes):
        thread = threads[i % len(threads)]
        messages = _build_sample_messages(system_prompt, thread["turns"], target_chars)
        response = chat_completion(
            messages=messages,
            model=model,
            timeout=CALIBRATION_CALL_TIMEOUT_S,
            extra_body={"max_tokens": CALIBRATION_MAX_TOKENS},
        )
        chars = sum(len(m["content"]) for m in messages)
        samples.append((chars, len(messages), response.prompt_tokens))
        print(f"  sample {i + 1}/{len(target_sizes)}: {chars} chars, {len(messages)} messages -> {response.prompt_tokens} tokens")
    return samples


# Train sizes span ~200 to ~35,000 characters; held-out sizes are interleaved at
# different points in that range so the reported error covers small and large prompts.
TRAIN_TARGET_SIZES = [250, 800, 1600, 2800, 4500, 6500, 9000, 13000, 17000, 22000, 28000, 35000]
HOLD_OUT_TARGET_SIZES = [500, 3500, 11000, 30000]


def main() -> None:
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    thread_paths = sorted(Path("data/threads").glob("*.json"))
    threads = [json.loads(p.read_text()) for p in thread_paths]
    if not threads:
        raise SystemExit("No threads in data/threads/, run generate_threads first")

    print(f"Collecting {len(TRAIN_TARGET_SIZES)} training samples...")
    train_samples = _collect_samples(model, threads, TRAIN_TARGET_SIZES)
    print(f"Collecting {len(HOLD_OUT_TARGET_SIZES)} held-out samples...")
    held_out_samples = _collect_samples(model, threads, HOLD_OUT_TARGET_SIZES)

    fit = fit_tokens_model(train_samples)
    error_pct = held_out_error(fit, held_out_samples)

    out_path = Path("data/calibration.json")
    out_path.write_text(
        json.dumps(
            {
                "a": fit.a,
                "b": fit.b,
                "c": fit.c,
                "n_train": len(train_samples),
                "n_held_out": len(held_out_samples),
                "held_out_error_pct": error_pct,
                "model": model,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"\nFit: tokens = {fit.a:.4f}*chars + {fit.b:.2f}*messages + {fit.c:.2f}\n"
        f"Held-out mean absolute percentage error: {error_pct:.1f}% over {len(held_out_samples)} samples\n"
        f"Wrote {out_path}"
    )


if __name__ == "__main__":
    main()
