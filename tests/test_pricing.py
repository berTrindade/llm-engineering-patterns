from pathlib import Path

from cost_and_routing.pricing import ModelPrice, cost_usd, load_prices

TOML_FIXTURE = """
[models.cheap]
input_per_million = 0.10
output_per_million = 0.30

[models.strong]
input_per_million = 0.60
output_per_million = 2.00
"""


def test_load_prices_reads_models(tmp_path: Path) -> None:
    toml_path = tmp_path / "prices.toml"
    toml_path.write_text(TOML_FIXTURE)

    prices = load_prices(str(toml_path))

    assert prices["cheap"] == ModelPrice(input_per_million=0.10, output_per_million=0.30)
    assert prices["strong"] == ModelPrice(input_per_million=0.60, output_per_million=2.00)


def test_cost_usd_computes_input_and_output_cost() -> None:
    price = ModelPrice(input_per_million=1.0, output_per_million=2.0)

    cost = cost_usd(prompt_tokens=1_000_000, completion_tokens=500_000, price=price)

    assert cost == 1.0 + 1.0


def test_cost_usd_zero_tokens_is_zero() -> None:
    price = ModelPrice(input_per_million=5.0, output_per_million=5.0)

    assert cost_usd(prompt_tokens=0, completion_tokens=0, price=price) == 0.0
