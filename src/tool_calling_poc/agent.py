from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from tool_calling_poc.llm import LLMResponse
from tool_calling_poc.state import OfficeState
from tool_calling_poc.tools import call_tool, tool_specs_for_llm


@dataclass
class TraceStep:
    iteration: int
    tool_name: str | None
    arguments: dict | None
    result_or_error: str | None
    is_error: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class AgentResult:
    final_message: str | None
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0
    tool_call_count: int = 0
    recovered_error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    stopped_reason: Literal["final_message", "iteration_cap"] = "iteration_cap"


def default_system_prompt(state: OfficeState) -> str:
    return (
        "You are a scheduling assistant for Harbour Street Studio. "
        f"The current date and time is {state.now.isoformat()}. "
        "Use the tools to check room capacity, equipment and everyone's availability before "
        "booking. If a request cannot be satisfied within the rules, explain why instead of "
        "booking something that breaks them."
    )


def run_agent_loop(
    state: OfficeState,
    user_message: str,
    chat_fn: Callable[..., LLMResponse],
    model: str,
    max_iterations: int = 8,
    system_prompt: str | None = None,
) -> AgentResult:
    messages: list[dict] = [
        {"role": "system", "content": system_prompt or default_system_prompt(state)},
        {"role": "user", "content": user_message},
    ]
    tools = tool_specs_for_llm()

    result = AgentResult(final_message=None)

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        response = chat_fn(messages=messages, tools=tools, model=model)
        result.prompt_tokens += response.prompt_tokens
        result.completion_tokens += response.completion_tokens
        result.latency_ms += response.latency_ms

        if not response.tool_calls:
            result.final_message = response.content
            result.stopped_reason = "final_message"
            result.trace.append(
                TraceStep(
                    iteration=iteration,
                    tool_name=None,
                    arguments=None,
                    result_or_error=response.content,
                    is_error=False,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=response.latency_ms,
                )
            )
            break

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            result.tool_call_count += 1
            arguments: dict | None
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError as exc:
                arguments = None
                ok, result_text = False, f"could not parse tool arguments as JSON: {exc}"
            else:
                ok, result_text = call_tool(state, call.name, arguments)

            if not ok:
                result.recovered_error_count += 1

            result.trace.append(
                TraceStep(
                    iteration=iteration,
                    tool_name=call.name,
                    arguments=arguments,
                    result_or_error=result_text,
                    is_error=not ok,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_ms=response.latency_ms,
                )
            )

            messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})

    return result
