"""Fits the estimator's chars_per_token ratio against real API usage.

Run via `make calibrate`. Not covered by tests beyond fit_ratio itself: it makes real
calls to CHEAP_MODEL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from context_window_poc.llm import chat_completion

CALIBRATION_PROMPTS = [
    "Summarise in one sentence: the customer's invoice was overdue by ten days.",
    "List three reasons a support ticket might be reopened after it was closed.",
    "What time zone abbreviation is used for Central European Summer Time?",
    "Write a two-sentence apology for a delayed refund.",
    "State the account id ACC-4821 and plan name Starter back to the customer.",
]


def fit_ratio(samples: list[tuple[str, int]]) -> float:
    total_chars = sum(len(text) for text, _ in samples)
    total_tokens = sum(tokens for _, tokens in samples)
    return total_chars / total_tokens


def main() -> None:
    model = os.environ.get("CHEAP_MODEL", "qwen3:8b")
    samples: list[tuple[str, int]] = []
    for prompt in CALIBRATION_PROMPTS:
        response = chat_completion(messages=[{"role": "user", "content": prompt}], model=model)
        samples.append((prompt, response.prompt_tokens))

    ratio = fit_ratio(samples)
    out_path = Path("data/calibration.json")
    out_path.write_text(
        json.dumps(
            {
                "chars_per_token": ratio,
                "n_samples": len(samples),
                "model": model,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Calibrated chars_per_token = {ratio:.3f} from {len(samples)} prompts, wrote {out_path}")


if __name__ == "__main__":
    main()
