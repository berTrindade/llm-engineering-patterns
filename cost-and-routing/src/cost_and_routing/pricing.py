"""Price table lookup and cost arithmetic.

Prices are illustrative USD per million tokens, loaded from prices.toml. Local Ollama
models have no real bill; the numbers here only show what the recorded token counts
would cost against those illustrative rates.
"""

import tomllib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


def load_prices(path: str = "prices.toml") -> dict[str, ModelPrice]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return {
        name: ModelPrice(
            input_per_million=values["input_per_million"],
            output_per_million=values["output_per_million"],
        )
        for name, values in raw["models"].items()
    }


def cost_usd(prompt_tokens: int, completion_tokens: int, price: ModelPrice) -> float:
    input_cost = (prompt_tokens / 1_000_000) * price.input_per_million
    output_cost = (completion_tokens / 1_000_000) * price.output_per_million
    return input_cost + output_cost
