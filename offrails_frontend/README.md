# OffRails 🛤️
**Agent Trace Anomaly Detection**

Paste an agent trace JSON → visualize the flow → get an anomaly score and behavioral flags.

## Quick Start

```bash
npm install
npm start
```

Opens at `http://localhost:3000`

## Project Structure

```
src/
├── App.jsx                    # Main layout, 3-panel UI
├── App.module.css
├── components/
│   ├── ScoreRing.jsx          # Animated SVG anomaly score ring
│   ├── ScoreRing.module.css
│   ├── FlowDiagram.jsx        # Step-by-step trace visualizer
│   └── FlowDiagram.module.css
└── utils/
    └── traceAnalysis.js       # Parsing + rule-based scoring (swap for model API)
```

## Connecting Your Model

In `src/utils/traceAnalysis.js`, replace `analyzeTrace()` with a call to your FastAPI backend:

```js
export async function analyzeTrace(steps) {
  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steps }),
  });
  return res.json(); // { score, flags, metrics }
}
```

## Trace Format

```json
{
  "conversations": [
    { "from": "system",    "value": "You are..." },
    { "from": "user",      "value": "What is..." },
    { "from": "assistant", "value": "I'll look that up.", "tool_calls": [{"name": "web_search"}] },
    { "from": "tool",      "value": "Result..." },
    { "from": "assistant", "value": "The answer is..." }
  ]
}
```

Also accepts `messages` array or a bare array of steps.
