"""Tool definitions for the tutor agent.

Each tool is a thin wrapper over an HTTP endpoint on the Rust MLP server.
`TOOL_SCHEMAS` is what we advertise to the LLM (OpenAI function-calling format,
which OpenRouter accepts verbatim); `dispatch` executes a tool call.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

MLP_URL = os.environ.get("MLP_URL", "http://127.0.0.1:9000")

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Read the current state of the MLP: layer sizes, activation functions, all "
                "weights and biases, the learning rate, the step counter, and the last "
                "training loss. Use this before giving advice so you're grounded in actual "
                "numbers."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "step",
            "description": (
                "Run N SGD training steps on the XOR dataset (cycling through the 4 "
                "examples). Returns the last loss and average loss across the batch. Use "
                "small N (10-200) to probe, larger N (500-5000) when confident the config "
                "will converge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10000,
                        "description": "Number of SGD steps to run.",
                    }
                },
                "required": ["count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_learning_rate",
            "description": (
                "Change the learning rate. The network keeps its current weights; only the "
                "LR changes. Reasonable range for this toy MLP is 0.01 to 5.0."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lr": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
                },
                "required": ["lr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "randomize",
            "description": (
                "Re-initialize the network with fresh weights. Step counter resets to 0. "
                "Pass a seed for reproducibility; omit it for truly random init."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seed": {
                        "type": "integer",
                        "description": "Optional seed for deterministic init.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forward",
            "description": (
                "Run the current network forward on a single input and return the output "
                "plus per-layer activations. Useful for checking what the network predicts "
                "for a specific XOR case without training."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    }
                },
                "required": ["inputs"],
            },
        },
    },
]


async def dispatch(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Execute a named tool call by proxying to the Rust server."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        if name == "get_state":
            r = await client.get(f"{MLP_URL}/state")
        elif name == "step":
            r = await client.post(f"{MLP_URL}/step", json={"count": tool_input.get("count", 1)})
        elif name == "set_learning_rate":
            r = await client.post(f"{MLP_URL}/set_lr", json={"lr": tool_input["lr"]})
        elif name == "randomize":
            body: dict[str, Any] = {}
            if "seed" in tool_input and tool_input["seed"] is not None:
                body["seed"] = tool_input["seed"]
            r = await client.post(f"{MLP_URL}/randomize", json=body)
        elif name == "forward":
            r = await client.post(
                f"{MLP_URL}/forward", json={"inputs": tool_input["inputs"]}
            )
        else:
            return {"error": f"unknown tool: {name}"}
        if r.status_code >= 400:
            return {"error": f"{r.status_code}: {r.text}"}
        return r.json()
