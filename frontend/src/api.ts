import type { AgentAnswer, ForwardResult, NetState, StepResult } from './mlp';

// Single-origin design: browser -> backend (:8000 in dev via Vite proxy, same origin in prod).
// Backend proxies /mlp/* to the internal Rust server.
const BASE = import.meta.env.VITE_API_BASE ?? '';

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}: ${await res.text()}`);
  }
  return res.json() as Promise<T>;
}

export const mlpApi = {
  state: () => jsonFetch<NetState>(`${BASE}/mlp/state`),
  step: (count: number) =>
    jsonFetch<StepResult>(`${BASE}/mlp/step`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ count }),
    }),
  randomize: (seed?: number) =>
    jsonFetch<NetState>(`${BASE}/mlp/randomize`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(seed === undefined ? {} : { seed }),
    }),
  setLr: (lr: number) =>
    jsonFetch<NetState>(`${BASE}/mlp/set_lr`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ lr }),
    }),
  forward: (inputs: number[]) =>
    jsonFetch<ForwardResult>(`${BASE}/mlp/forward`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ inputs }),
    }),
  healthz: () => jsonFetch<{ status: string }>(`${BASE}/mlp/healthz`),
};

export const agentApi = {
  ask: (question: string) =>
    jsonFetch<AgentAnswer>(`${BASE}/agent/ask`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ question }),
    }),
  healthz: () => jsonFetch<{ status: string }>(`${BASE}/healthz`),
};
