# MLP Trainer + LLM Tutor

A tiny multi-layer perceptron, implemented from scratch in Rust, that you can watch train on XOR in real time — paired with a Claude agent that observes the same state you do and can intervene (adjust the learning rate, step training, re-initialize weights) through tool-calling.

Started life as a Rust-only MLP simulator; grew an AI-engineer layer on top so the training UI is driven by a real agentic LLM loop.

## What it does

- **Rust MLP core** — forward pass, backprop, MSE loss, sigmoid activations. No ML libraries; the matrix math is written by hand in `src-tauri/src/mlp/network.rs`.
- **HTTP control plane (axum)** — `GET /state`, `POST /step`, `POST /randomize`, `POST /set_lr`, `POST /forward`.
- **FastAPI agent sidecar** — exposes `/agent/ask`. Every request runs an OpenRouter tool-calling loop (default model `openai/gpt-4o-mini`, any tool-capable OpenRouter model works via `OPENROUTER_MODEL`) with five tools mapped to the Rust endpoints: `get_state`, `step`, `set_learning_rate`, `randomize`, `forward`. The agent can diagnose, explain, *and* act.
- **React frontend** — left pane: live neuron viz, training controls, loss sparkline, per-example prediction table. Right pane: chat with the tutor, with every tool call surfaced as a collapsible trace.
- **Single-origin deploy** — the FastAPI backend proxies `/mlp/*` to the internal Rust server and serves the frontend bundle. One Dockerfile, one Fly.io process.

## Architecture

```
Browser
  │  (same origin)
  ▼
FastAPI (:8080)            ← public
  ├─ /healthz
  ├─ /agent/ask            ← Claude tool-calling loop
  ├─ /mlp/*                ← proxy to Rust
  └─ /                     ← static frontend bundle
             │
             ▼
   Rust axum (127.0.0.1:9000)   ← internal only
     forward / backprop / SGD
```

## Running it locally

Three terminals for dev (so you get Vite HMR and fast Rust rebuilds):

```sh
# 1. Rust MLP server
cd src-tauri && cargo run --release

# 2. Python agent sidecar
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
OPENROUTER_API_KEY=sk-or-v1-... uvicorn app.main:app --port 8000

# 3. Frontend (proxies /mlp and /agent to the backend)
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173.

## One-command deploy

```sh
# First time
fly launch --copy-config --no-deploy    # edit `app =` to a unique name
fly secrets set OPENROUTER_API_KEY=sk-or-v1-...
# optional:  fly secrets set OPENROUTER_MODEL=anthropic/claude-sonnet-4.5
fly deploy
```

The `Dockerfile` is a three-stage build: Rust → frontend → Python runtime. The runtime image carries only the Rust binary, the Python `app/` module, and the static frontend bundle.

## How the agent is wired

`backend/app/agent.py` is the full loop. Each iteration:

1. Send the conversation to the OpenRouter-routed model with the five tools attached (OpenAI function-calling format).
2. If the model returns no `tool_calls`, return the final text.
3. Otherwise, execute each requested tool by HTTP-calling the Rust server, feed the JSON results back as `role: "tool"` messages, and loop.

Results are summarized (weights/biases stripped) before being shown in the UI trace, but the full results go back to the model so it can keep reasoning over real numbers.

## Skills matrix

| Feature | What it demonstrates |
|---|---|
| OpenRouter-routed tool-calling agent loop with structured inputs | LLM engineering — agents, function calling, structured outputs, provider abstraction |
| FastAPI service with a typed proxy + response models | API design, production Python |
| Rust `axum` + `tokio` server, handwritten MLP | Systems Rust, ML fundamentals from first principles |
| Multi-stage Dockerfile, single-origin serve, Fly.io config | Containerized deploy, production shape |
| Agent returns trace of actual tool calls made | Observability into agent behavior |
| Frontend shows live network state while the agent mutates it | End-to-end systems thinking |

## What this project does **not** cover

Being honest so recruiters aren't surprised:

- **No RAG.** Tracked for a separate portfolio project (agentic RAG over financial docs).
- **No multi-agent orchestration.** Single-agent design here; LangGraph / CrewAI work lives elsewhere.
- **No fine-tuned model.** The MLP is trained live; the LLM is called via API. Separate project for PyTorch fine-tuning.
- **No fintech domain content.** XOR is the demo dataset.
- **Training is blocking on the Rust server** (single mutex). Fine for one user watching one session; not a production MLOps story.

## Layout

```
src-tauri/        Rust MLP server (axum)
  src/mlp/        network.rs (forward/backprop), activations.rs
  src/main.rs     HTTP routes + AppState
backend/
  app/
    main.py       FastAPI routes + MLP proxy + static mount
    agent.py      OpenRouter tool-calling loop
    tools.py      tool schemas + HTTP dispatch
frontend/
  src/App.tsx     two-pane UI
  src/viz/NeuronViz.tsx  live network SVG, generalized to any layer_sizes
docker/entrypoint.sh    runtime process supervisor
Dockerfile        three-stage build
fly.toml          Fly.io deploy config
```

## License

MIT.
