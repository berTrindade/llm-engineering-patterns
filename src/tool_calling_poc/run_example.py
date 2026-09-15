from __future__ import annotations

from pathlib import Path

from tool_calling_poc.agent import run_agent_loop
from tool_calling_poc.config import load_config
from tool_calling_poc.llm import chat_completion
from tool_calling_poc.state import OfficeState

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "office.json"

EXAMPLE_REQUEST = (
    "Book a 30 minute meeting for Alice Byrne and Ben Ochieng in the Kittiwake room "
    "on Wednesday 16 September 2026 at 11:00 UTC, titled 'Planning sync'."
)


def main() -> None:
    config = load_config()
    state = OfficeState.from_json(DATA_PATH)

    def chat_fn(**kwargs):
        return chat_completion(base_url=config.llm_base_url, api_key=config.llm_api_key, **kwargs)

    result = run_agent_loop(state, EXAMPLE_REQUEST, chat_fn, model=config.cheap_model)

    print(f"Request: {EXAMPLE_REQUEST}\n")
    for step in result.trace:
        if step.tool_name:
            status = "error" if step.is_error else "ok"
            print(
                f"[{step.iteration}] {step.tool_name}({step.arguments}) "
                f"-> {status}: {step.result_or_error}"
            )
    print(f"\nFinal message: {result.final_message}")
    print(
        f"Stopped: {result.stopped_reason}, iterations: {result.iterations}, "
        f"tool calls: {result.tool_call_count}"
    )


if __name__ == "__main__":
    main()
