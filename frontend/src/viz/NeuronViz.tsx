import React from 'react';
import type { NetState } from '../mlp';

type Props = {
  state: NetState;
  training: boolean;
};

const W = 560, H = 240;
const BAND_TOP = 24, BAND_BOTTOM = 200;
const COLS = [100, 280, 460];

const LAYER_FRACS = [
  [0.25, 0.75],
  [0.10, 0.37, 0.63, 0.90],
  [0.50],
];

function yAt(f: number) {
  return BAND_TOP + f * (BAND_BOTTOM - BAND_TOP);
}

const POSITIONS = LAYER_FRACS.map((fracs, li) =>
  fracs.map(f => ({ x: COLS[li], y: yAt(f) }))
);

const NeuronViz: React.FC<Props> = ({ state, training }) => {
  const { weights } = state;

  const edges: { x1: number; y1: number; x2: number; y2: number; w: number }[] = [];
  for (let l = 0; l < POSITIONS.length - 1; l++) {
    for (let a = 0; a < POSITIONS[l].length; a++) {
      for (let b = 0; b < POSITIONS[l + 1].length; b++) {
        const w = weights[l]?.[b]?.[a] ?? 0;
        edges.push({
          x1: POSITIONS[l][a].x, y1: POSITIONS[l][a].y,
          x2: POSITIONS[l + 1][b].x, y2: POSITIONS[l + 1][b].y,
          w,
        });
      }
    }
  }

  return (
    <div className="net-stage">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {edges.map((e, i) => {
          const mag = Math.min(1, Math.abs(e.w));
          return (
            <line
              key={i}
              x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              stroke={e.w >= 0 ? 'var(--accent)' : 'var(--neg)'}
              strokeWidth={0.4 + mag * 2.2}
              strokeOpacity={0.15 + mag * 0.7}
              strokeLinecap="round"
            />
          );
        })}
        {POSITIONS.flat().map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r={13} fill="var(--paper)" stroke="var(--ink)" strokeWidth={1.5} />
            <circle cx={p.x} cy={p.y} r={4} fill="var(--ink)" opacity={training ? 0.9 : 0.6}>
              {training && (
                <animate attributeName="r" values="3;5;3" dur="1.2s" repeatCount="indefinite" />
              )}
            </circle>
          </g>
        ))}
        <g fontFamily="var(--mono)" fontSize="10" fill="oklch(0.60 0.01 270)" textAnchor="middle">
          <text x={COLS[0]} y={H - 8}>input · 2</text>
          <text x={COLS[1]} y={H - 8}>hidden · 4</text>
          <text x={COLS[2]} y={H - 8}>output · 1</text>
        </g>
      </svg>
      <div className="net-chip">XOR · 2–4–1 · σ</div>
      <div className="net-legend">
        <span><i className="swatch pos" /> positive weight</span>
        <span><i className="swatch neg" /> negative weight</span>
      </div>
    </div>
  );
};

export default NeuronViz;
