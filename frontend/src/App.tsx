import { useCallback, useEffect, useRef, useState } from 'react';
import { agentApi, mlpApi } from './api';
import type { AgentTrace, NetState } from './mlp';
import NeuronViz from './viz/NeuronViz';
import './App.css';

type Busy = 'idle' | 'training' | 'asking';

type ChatMessage =
  | { role: 'user'; text: string }
  | { role: 'assistant'; text: string; trace: AgentTrace[] };

const SUGGESTIONS = [
  { kind: 'read', label: 'Inspect',  text: 'What is the current state of the network?' },
  { kind: 'diag', label: 'Diagnose', text: 'Why is the loss not going down?' },
  { kind: 'act',  label: 'Act',      text: 'Reseed 42, train 500 steps, report back.' },
  { kind: 'read', label: 'Verify',   text: 'Check whether the network has solved XOR yet.' },
];

function SeedField({
  value, onChange, disabled,
}: { value: number; onChange: (v: number) => void; disabled: boolean }) {
  return (
    <div className="seed-field">
      <span className="seed-lbl">seed</span>
      <input
        type="number" min={0} step={1} value={value} disabled={disabled}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === '') { onChange(0); return; }
          onChange(Math.max(0, Math.floor(Number(raw) || 0)));
        }}
        onKeyDown={(e) => {
          if (e.key === '-' || e.key === '+' || e.key === 'e') e.preventDefault();
        }}
      />
      <div className="seed-steppers">
        <button onClick={() => onChange(value + 1)} disabled={disabled} aria-label="increment">▲</button>
        <button onClick={() => onChange(Math.max(0, value - 1))} disabled={disabled} aria-label="decrement">▼</button>
      </div>
    </div>
  );
}

function Stepper({
  onStep, disabled, activeN,
}: { onStep: (n: number) => void; disabled: boolean; activeN: number | null }) {
  const opts = [1, 100, 1000, 10000];
  return (
    <div className="stepper">
      <span className="stepper-lead">step</span>
      {opts.map(n => (
        <button
          key={n} onClick={() => onStep(n)} disabled={disabled}
          className={activeN === n ? 'active' : ''}
        >
          ×{n.toLocaleString()}
        </button>
      ))}
    </div>
  );
}

function LossChart({ losses }: { losses: number[] }) {
  if (losses.length < 2) {
    return (
      <div className="loss-chart">
        <div className="loss-chart-empty">run a few steps to see loss</div>
      </div>
    );
  }
  const W = 600, H = 72, pad = 6;
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = Math.max(0.0001, max - min);
  const pts = losses.map((v, i): [number, number] => [
    pad + (i / (losses.length - 1)) * (W - pad * 2),
    pad + (1 - (v - min) / range) * (H - pad * 2),
  ]);
  const d = 'M ' + pts.map(p => p.join(',')).join(' L ');
  const area = `${d} L ${pts[pts.length - 1][0]},${H} L ${pts[0][0]},${H} Z`;
  const last = pts[pts.length - 1];
  return (
    <div className="loss-chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        <path d={area} fill="var(--accent-soft)" />
        <path d={d} fill="none" stroke="var(--accent)" strokeWidth="1.5" />
        <circle cx={last[0]} cy={last[1]} r="2.5" fill="var(--accent)" />
      </svg>
    </div>
  );
}

function EvalTable({
  dataset, predictions,
}: { dataset: [number[], number[]][]; predictions: number[] }) {
  return (
    <div className="eval">
      <div className="eval-head">
        <span>input</span><span>target</span><span>prediction</span><span>match</span>
      </div>
      {dataset.map(([inp, tgt], i) => {
        const p = predictions[i];
        const t = tgt[0];
        const ok = p !== undefined && Math.abs(p - t) < 0.15;
        const cls = p !== undefined && Math.round(p) === Math.round(t) ? 'good' : 'bad';
        return (
          <div key={i} className={`eval-row ${cls}`}>
            <span>[{inp.join(', ')}]</span>
            <span>{t}</span>
            <span className="eval-pred">
              <span>{p !== undefined ? p.toFixed(3) : '…'}</span>
              {p !== undefined && (
                <span className="eval-bar"><i style={{ width: `${p * 100}%` }} /></span>
              )}
            </span>
            <span className={ok ? 'eval-ok' : 'eval-no'}>{ok ? '✓' : '·'}</span>
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<NetState | null>(null);
  const [predictions, setPredictions] = useState<number[]>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [busy, setBusy] = useState<Busy>('idle');
  const [error, setError] = useState<string | null>(null);

  const [committedSeed, setCommittedSeed] = useState(0);
  const [pendingSeed, setPendingSeed] = useState(0);
  const [lrDraft, setLrDraft] = useState(0.5);
  const [activeN, setActiveN] = useState<number | null>(null);
  const lrTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [question, setQuestion] = useState('');
  const [chat, setChat] = useState<ChatMessage[]>([]);
  const chatRef = useRef<HTMLDivElement>(null);

  const refreshPredictions = useCallback(async (ds: [number[], number[]][]) => {
    const preds: number[] = [];
    for (const [input] of ds) {
      const { outputs } = await mlpApi.forward(input);
      preds.push(outputs[0]);
    }
    setPredictions(preds);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const s = await mlpApi.state();
      setState(s);
      setLrDraft(s.learning_rate);
      await refreshPredictions(s.dataset);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [refreshPredictions]);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [chat, busy]);

  async function initialize(seed: number) {
    if (busy !== 'idle') return;
    setBusy('training');
    setError(null);
    try {
      await mlpApi.randomize(seed);
      setCommittedSeed(seed);
      setPendingSeed(seed);
      setLossHistory([]);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy('idle');
    }
  }

  async function runSteps(count: number) {
    if (busy !== 'idle') return;
    setBusy('training');
    setActiveN(count);
    setError(null);
    try {
      const chunks = count <= 100 ? 1 : count <= 1000 ? 4 : 10;
      const chunkSize = Math.ceil(count / chunks);
      for (let c = 0; c < chunks; c++) {
        const n = Math.min(chunkSize, count - c * chunkSize);
        if (n <= 0) break;
        const res = await mlpApi.step(n);
        setLossHistory(prev => [...prev, res.last_loss].slice(-400));
        await refresh();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy('idle');
      setActiveN(null);
    }
  }

  function onLrChange(v: number) {
    setLrDraft(v);
    if (lrTimer.current) clearTimeout(lrTimer.current);
    lrTimer.current = setTimeout(async () => {
      try { await mlpApi.setLr(v); } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }, 200);
  }

  async function ask(q: string) {
    if (busy !== 'idle' || !q.trim()) return;
    setBusy('asking');
    setError(null);
    setChat(c => [...c, { role: 'user', text: q }]);
    setQuestion('');
    try {
      const answer = await agentApi.ask(q);
      setChat(c => [...c, { role: 'assistant', text: answer.answer, trace: answer.trace }]);
      await refresh();
    } catch (e) {
      setChat(c => [...c, {
        role: 'assistant',
        text: `[error] ${e instanceof Error ? e.message : String(e)}`,
        trace: [],
      }]);
    } finally {
      setBusy('idle');
    }
  }

  const lastLoss = lossHistory.length ? lossHistory[lossHistory.length - 1] : null;
  const firstLoss = lossHistory.length ? lossHistory[0] : null;
  const delta = lastLoss !== null && firstLoss !== null ? lastLoss - firstLoss : null;
  const training = busy === 'training';
  const asking = busy === 'asking';

  return (
    <div className="app">
      {/* TOP BAR */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true" />
          <h1 className="title">MLP Trainer <span className="dim">· with an LLM tutor</span></h1>
        </div>
        <div className="status-pills">
          <span className="pill">
            <i className={`dot ${training ? 'train' : state && state.step_count > 0 ? '' : 'idle'}`} />
            {training ? 'training' : state && state.step_count > 0 ? 'paused' : 'idle'}
          </span>
          <span className="pill">step <b>{(state?.step_count ?? 0).toLocaleString()}</b></span>
          <span className="pill">loss <b>{lastLoss !== null ? lastLoss.toFixed(4) : '—'}</b></span>
        </div>
      </header>

      <div className="grid">
        {/* TRAINING CARD */}
        <section className="card">
          <div className="card-head">
            <h2>Training</h2>
            <div className="meta">
              <span>arch <b>2–4–1</b></span>
              <span>task <b>XOR</b></span>
              <span>seed <b>{committedSeed}</b></span>
            </div>
          </div>

          {state && <NeuronViz state={state} training={training} />}

          <div className="controls">
            {/* LR slider */}
            <div className="lr-row">
              <label>learning rate</label>
              <input
                type="range" min={0.01} max={2} step={0.01} value={lrDraft}
                onChange={(e) => onLrChange(Number(e.target.value))}
              />
              <output>{lrDraft.toFixed(3)}</output>
            </div>

            {/* Initialize + Seed | Reset */}
            <div className="bar-row">
              <div className="bar-left">
                <button
                  className="btn primary"
                  onClick={() => initialize(pendingSeed)}
                  disabled={busy !== 'idle'}
                >
                  Initialize
                </button>
                <SeedField value={pendingSeed} onChange={setPendingSeed} disabled={busy !== 'idle'} />
                {pendingSeed !== committedSeed && (
                  <span className="dirty-warn">· unapplied seed</span>
                )}
              </div>
              <div className="bar-right">
                <button className="btn ghost" onClick={() => initialize(committedSeed)} disabled={busy !== 'idle'}>
                  ↺ Reset
                </button>
              </div>
            </div>

            {/* Step buttons */}
            <div className="bar-row">
              <Stepper onStep={runSteps} disabled={busy !== 'idle'} activeN={activeN} />
              {training && (
                <span className="running-indicator">
                  <i className="dot train" />
                  running…
                </span>
              )}
            </div>
          </div>

          {/* Loss */}
          <div className="loss-block">
            <div className="loss-head">
              <span className="loss-label">recent loss</span>
              <span className="loss-value">
                {lastLoss !== null ? lastLoss.toFixed(4) : '—'}
                {delta !== null && (
                  <span className="loss-delta" style={{ color: delta < 0 ? 'var(--pos)' : 'var(--neg)' }}>
                    {delta < 0 ? ' ▾' : ' ▴'} {Math.abs(delta).toFixed(4)}
                  </span>
                )}
              </span>
            </div>
            <LossChart losses={lossHistory} />
          </div>

          {/* Eval table */}
          {state && <EvalTable dataset={state.dataset} predictions={predictions} />}
        </section>

        {/* TUTOR CARD */}
        <section className="card tutor-card">
          <div className="card-head">
            <h2>Tutor</h2>
            <div className="meta">
              <span className="pill"><i className="dot" /> connected</span>
            </div>
          </div>

          <div className="tutor-body" ref={chatRef}>
            {chat.length === 0 && (
              <>
                <div className="tutor-intro">
                  Ask the tutor about the network. It can <b>read live state</b>, <b>diagnose</b> slow
                  training, and <b>take actions</b> like reseeding or stepping on your behalf.
                </div>
                <div className="chips-label">Suggestions</div>
                <div className="chips">
                  {SUGGESTIONS.map((s, i) => (
                    <button key={i} className={`chip ${s.kind}`} onClick={() => ask(s.text)}>
                      <span className="chip-k">{s.label}</span>
                      <span className="chip-t">{s.text}</span>
                    </button>
                  ))}
                </div>
              </>
            )}
            {chat.map((m, i) => (
              <div key={i} className={`msg msg-${m.role}`}>
                {m.role === 'user' ? (
                  <div className="bubble-user">{m.text}</div>
                ) : (
                  <>
                    <div className="msg-who">Tutor · llm</div>
                    {m.trace.map((t, j) => (
                      <div key={j} className="tool-block">
                        <div className="tool-call">→ {t.tool}({JSON.stringify(t.input)})</div>
                        <div className="tool-result">← {JSON.stringify(t.result).slice(0, 200)}</div>
                      </div>
                    ))}
                    <div className="bubble-assistant">{m.text}</div>
                  </>
                )}
              </div>
            ))}
            {asking && (
              <div className="msg">
                <div className="msg-who">Tutor · llm</div>
                <div className="bubble-assistant" style={{ color: 'var(--ink-3)' }}>thinking…</div>
              </div>
            )}
          </div>

          <div className="composer">
            <div className="composer-row">
              <textarea
                placeholder="Ask the tutor anything about this network…"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(question); }
                }}
                disabled={busy !== 'idle'}
                maxLength={2000}
              />
              <button
                className="btn accent"
                onClick={() => ask(question)}
                disabled={busy !== 'idle' || !question.trim()}
              >
                Ask <span className="kbd">⏎</span>
              </button>
            </div>
            <div className="composer-foot">
              <div className="composer-foot-left">
                <span className="model-pill">llm · tool-calling</span>
                <span>can read state · can step · can reseed</span>
              </div>
              <div>{question.length} / 2000</div>
            </div>
          </div>
        </section>
      </div>

      {error && <div className="toast">{error}</div>}

    </div>
  );
}
