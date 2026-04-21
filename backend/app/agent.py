"""Tutor agent — tool-calling loop against OpenRouter (OpenAI-compatible API).

Takes a user question, gives the model access to the MLP tools, lets it call as
many as it needs, returns the final answer plus a structured trace of tool
calls so the frontend can show what the agent actually did.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from .tools import TOOL_SCHEMAS, dispatch

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
REFERER = os.environ.get("OPENROUTER_REFERER", "https://github.com/mlp-tutor")
APP_TITLE = os.environ.get("OPENROUTER_APP_TITLE", "MLP Tutor")

MAX_ITERATIONS = 10
MAX_TOKENS = 2048

SYSTEM_PROMPT = """\
You are a neural-network tutor watching a tiny multi-layer perceptron train on XOR.
The network is a 2-4-1 MLP with sigmoid activations, implemented in Rust.

You can observe the network's live state and intervene via tools:
- get_state: inspect weights, biases, learning rate, step count, last loss
- step(count): run N SGD steps
- set_learning_rate(lr): adjust the LR
- randomize(seed?): reset weights
- forward(inputs): predict for one input without training

Style:
- Always call get_state before diagnosing or advising — ground your answers in actual numbers.
- When the user asks you to fix or improve something, feel free to act (step, set_lr, randomize)
  rather than just suggesting. A few small interventions beat a wall of text.
- When explaining, refer to concrete numbers ("your output-layer weights are all near 0.1,
  so the net is under-confident") rather than generic advice.
- Keep answers tight. Bullet points or 3-4 short sentences.
- XOR is the canonical non-linearly-separable toy: confirm the network really is solving
  it (predictions near 0, 1, 1, 0) before declaring success.
"""


class AgentResult(dict):
    """Structured result: answer string + trace of tool calls."""


async def run_agent(question: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
    if not os.environ.get("OPENROUTER_API_KEY"):
        return AgentResult(
            answer=(
                "OPENROUTER_API_KEY is not set on the backend. "
                "Set it in the environment and restart the server."
            ),
            trace=[],
            error="missing_api_key",
        )

    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        default_headers={
            # OpenRouter uses these for analytics / free-tier rankings; both optional.
            "HTTP-Referer": REFERER,
            "X-Title": APP_TITLE,
        },
    )

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    trace: list[dict[str, Any]] = []

    for _ in range(MAX_ITERATIONS):
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        choice = response.choices[0]
        msg = choice.message

        assistant_turn: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            return AgentResult(
                answer=(msg.content or "").strip() or "(no text returned)",
                trace=trace,
                stop_reason=choice.finish_reason,
            )

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await dispatch(tc.function.name, args)
            trace.append(
                {
                    "tool": tc.function.name,
                    "input": args,
                    "result": _summarize_for_trace(result),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, separators=(",", ":")),
                }
            )

    return AgentResult(
        answer="(agent hit max iterations without finishing)",
        trace=trace,
        stop_reason="max_iterations",
    )


def _summarize_for_trace(result: dict[str, Any]) -> dict[str, Any]:
    """Slim result for the UI trace — strip bulky weights/biases so the UI chip is readable."""
    if "error" in result:
        return {"error": result["error"]}
    slim = {k: v for k, v in result.items() if k not in ("weights", "biases", "dataset")}
    if "weights" in result:
        slim["weights_shape"] = [
            [len(result["weights"][i]), len(result["weights"][i][0])]
            for i in range(len(result["weights"]))
        ]
    return slim
