import React, { useMemo, useState } from 'react';

const ROLE_COLORS = {
  system:    { fill: '#F1EFE8', stroke: '#B4B2A9', text: '#444441' },
  user:      { fill: '#EEEDFE', stroke: '#AFA9EC', text: '#3C3489' },
  assistant: { fill: '#FBEAF0', stroke: '#ED93B1', text: '#72243E' },
  tool:      { fill: '#E1F5EE', stroke: '#5DCAA5', text: '#085041' },
  result:    { fill: '#F1EFE8', stroke: '#B4B2A9', text: '#444441' },
};

const LEGEND = [
  { label: 'user',        ...ROLE_COLORS.user },
  { label: 'assistant',   ...ROLE_COLORS.assistant },
  { label: 'tool call',   ...ROLE_COLORS.tool },
  { label: 'observation', ...ROLE_COLORS.result },
];

const NODE_W = 212;
const NODE_H = 44;
const NODE_X = 234;
const CENTER_X = NODE_X + NODE_W / 2;
const ROW_GAP = 72;
const START_Y = 50;

function truncate(str, max = 28) {
  if (!str || typeof str !== 'string') return '';
  return str.length > max ? str.slice(0, max - 1) + '…' : str;
}

// Map "from" field to one of our 5 role keys
function getRole(step) {
  const r = (step.from || step.role || '').toLowerCase();
  if (r === 'tool_call' || r === 'function_call') return 'tool';
  if (r === 'observation' || r === 'tool_response' || r === 'function') return 'result';
  if (r === 'system') return 'system';
  if (r === 'user') return 'user';
  if (r === 'assistant' || r === 'gpt' || r === 'chatgpt') return 'assistant';
  return 'assistant';
}

function getText(step) {
  const raw = step.value || step.content || '';
  return typeof raw === 'string' ? raw : JSON.stringify(raw);
}

function buildRows(steps) {
  const rows = [];
  let y = START_Y;
  steps.forEach((step, i) => {
    const role = getRole(step);
    const text = getText(step);
    rows.push({ type: 'node', step, role, text, y, index: i });
    y += ROW_GAP;
  });
  return rows;
}

function NodeRect({ x, y, w, h, role, label, sub, anomaly }) {
  const c = ROLE_COLORS[role] || ROLE_COLORS.result;
  return (
    <g>
      <rect
        x={x} y={y} width={w} height={h} rx={8}
        fill={c.fill}
        stroke={anomaly ? '#c49040' : c.stroke}
        strokeWidth={anomaly ? 1.5 : 0.5}
      />
      {anomaly && (
        <rect
          x={x - 3} y={y - 3} width={w + 6} height={h + 6} rx={11}
          fill="none" stroke="#c49040" strokeWidth={1} strokeDasharray="5 3" opacity={0.6}
        />
      )}
      {sub ? (
        <>
          <text
            x={x + w / 2} y={y + h / 2 - 8}
            textAnchor="middle" dominantBaseline="central"
            style={{ fill: c.text, fontSize: 13, fontWeight: 600,
              fontFamily: '-apple-system, SF Pro Display, sans-serif' }}
          >
            {label}
          </text>
          <text
            x={x + w / 2} y={y + h / 2 + 10}
            textAnchor="middle" dominantBaseline="central"
            style={{ fill: c.text, fontSize: 11, opacity: 0.75,
              fontFamily: '-apple-system, SF Pro Text, sans-serif' }}
          >
            {sub}
          </text>
        </>
      ) : (
        <text
          x={x + w / 2} y={y + h / 2}
          textAnchor="middle" dominantBaseline="central"
          style={{ fill: c.text, fontSize: 13, fontWeight: 600,
            fontFamily: '-apple-system, SF Pro Display, sans-serif' }}
        >
          {label}
        </text>
      )}
    </g>
  );
}

export default function FlowDiagram({ steps, flags = [] }) {
  const [zoom, setZoom] = useState(1);
  const rows = useMemo(() => buildRows(steps), [steps]);

  const hasCircular = flags.some(f => f.tag?.includes('circular'));
  const hasErrors   = flags.some(f => f.tag?.includes('observation'));
  const hasGiveUp   = flags.some(f => f.tag?.includes('goal'));

  // Mark anomalous nodes based on backend flags
  const isAnomalousNode = (role, text) => {
    if (hasCircular && role === 'tool') return true;
    if (hasErrors && role === 'result' && text.toLowerCase().includes('error')) return true;
    if (hasGiveUp && role === 'assistant' && (text.toLowerCase().includes('give up') || text.toLowerCase().includes("won't be able"))) return true;
    return false;
  };

  const toolCallCount = steps.filter(s => getRole(s) === 'tool').length;

  const totalH = rows.length > 0
    ? rows[rows.length - 1].y + NODE_H + 60
    : 200;
  const viewH = Math.max(totalH, 300);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6, marginBottom: 8 }}>
        <button onClick={() => setZoom(z => Math.max(0.5, +(z - 0.15).toFixed(2)))} style={btnStyle}>−</button>
        <span style={{ fontSize: 11, color: '#9b8a86', alignSelf: 'center', minWidth: 36, textAlign: 'center' }}>
          {Math.round(zoom * 100)}%
        </span>
        <button onClick={() => setZoom(z => Math.min(2, +(z + 0.15).toFixed(2)))} style={btnStyle}>+</button>
      </div>
      <div style={{ overflow: 'auto', flex: 1 }}>
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top center', transition: 'transform 0.15s ease' }}>
          <svg width="100%" viewBox={`0 0 500 ${viewH}`} style={{ display: 'block' }}>
            <defs>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
                  strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </marker>
            </defs>

            <text
              x={CENTER_X} y={22} textAnchor="middle"
              style={{ fill: '#9b8a86', fontSize: 11, letterSpacing: '0.07em',
                fontFamily: '-apple-system, SF Pro Text, sans-serif',
                textTransform: 'uppercase' }}
            >
              {steps.length} steps · {toolCallCount} tool calls
            </text>

            {rows.map((row, i) => {
              const { role, text, y } = row;
              const anomaly = isAnomalousNode(role, text);
              const sub = text ? truncate(text, 30) : null;
              const showSub = sub && role !== 'system';
              const h = showSub ? 52 : NODE_H;
              const nextRow = rows[i + 1];

              // Display label — use readable name for tool roles
              const displayLabel =
                role === 'tool'   ? 'tool call' :
                role === 'result' ? 'observation' :
                role;

              return (
                <g key={i}>
                  <NodeRect
                    x={NODE_X} y={y} w={NODE_W} h={h}
                    role={role}
                    label={displayLabel}
                    sub={showSub ? sub : null}
                    anomaly={anomaly}
                  />
                  {nextRow && (
                    <line
                      x1={CENTER_X} y1={y + h}
                      x2={CENTER_X} y2={nextRow.y}
                      stroke="#c2b0ab" strokeWidth={0.8}
                      markerEnd="url(#arrow)"
                    />
                  )}
                </g>
              );
            })}

            {/* Legend */}
            <g>
              <text
                x={44} y={START_Y + 2}
                style={{ fill: '#9b8a86', fontSize: 10, letterSpacing: '0.06em',
                  fontFamily: '-apple-system, SF Pro Text, sans-serif',
                  textTransform: 'uppercase' }}
              >
                legend
              </text>
              {LEGEND.map((l, i) => (
                <g key={l.label}>
                  <rect
                    x={40} y={START_Y + 16 + i * 28}
                    width={90} height={22} rx={6}
                    fill={l.fill} stroke={l.stroke} strokeWidth={0.5}
                  />
                  <text
                    x={85} y={START_Y + 16 + i * 28 + 11}
                    textAnchor="middle" dominantBaseline="central"
                    style={{ fill: l.text, fontSize: 11,
                      fontFamily: '-apple-system, SF Pro Text, sans-serif' }}
                  >
                    {l.label}
                  </text>
                </g>
              ))}
              <rect
                x={40} y={START_Y + 16 + LEGEND.length * 28}
                width={90} height={22} rx={6}
                fill="none" stroke="#c49040" strokeWidth={1} strokeDasharray="4 3"
              />
              <text
                x={85} y={START_Y + 16 + LEGEND.length * 28 + 11}
                textAnchor="middle" dominantBaseline="central"
                style={{ fill: '#c49040', fontSize: 11,
                  fontFamily: '-apple-system, SF Pro Text, sans-serif' }}
              >
                anomaly
              </text>
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}

const btnStyle = {
  width: 26, height: 26,
  border: '0.5px solid rgba(155,115,110,0.35)',
  borderRadius: 6,
  background: '#faf7f4',
  color: '#6b5a56',
  fontSize: 15,
  lineHeight: 1,
  cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontFamily: '-apple-system, SF Pro Text, sans-serif',
};