import React, { useState, useCallback } from 'react';
import styles from './App.module.css';
import ScoreRing from './components/ScoreRing';
import FlowDiagram from './components/FlowDiagram';
import { SAMPLE_TRACE_GOOD, SAMPLE_TRACE_BAD, parseTrace, analyzeTrace } from './utils/traceAnalysis';

function PanelHeader({ active, success, label, step, children }) {
  const dotClass = success ? styles.dotSuccess : active ? styles.dotActive : styles.dot;
  return (
    <div className={styles.panelHeader}>
      <div className={`${styles.dot} ${dotClass}`} />
      <span className={styles.panelLabel}>{label}</span>
      {children}
      <span className={styles.panelStep}>{step}</span>
    </div>
  );
}

function EmptyState({ icon, title, sub }) {
  return (
    <div className={styles.emptyState}>
      <div className={styles.emptyIcon}>{icon}</div>
      <span className={styles.emptyTitle}>{title}</span>
      <span className={styles.emptySub}>{sub}</span>
    </div>
  );
}

function Spinner() {
  return <div className={styles.spinner} />;
}

export default function App() {
  const [traceInput, setTraceInput] = useState('');
  const [steps, setSteps] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInvestigate = useCallback(async () => {
    setError('');
    const parsed = parseTrace(traceInput);
    if (!parsed) {
      setError('Invalid trace format. Please paste valid JSON.');
      return;
    }
    setLoading(true);
    setSteps(null);
    setResult(null);
    try {
      setSteps(parsed);
      const analysisResult = await analyzeTrace(parsed);
      setResult(analysisResult);
    } catch (err) {
      setError(`API error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [traceInput]);

  const loadSample = (which) => {
    setTraceInput(which === 'good' ? SAMPLE_TRACE_GOOD : SAMPLE_TRACE_BAD);
    setSteps(null);
    setResult(null);
    setError('');
  };

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
          <rect x="4" y="6" width="20" height="16" rx="3" fill="#e8dde8" stroke="#b89ab8" strokeWidth="0.8" />
          <line x1="8" y1="10" x2="20" y2="10" stroke="#9a759a" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="8" y1="14" x2="20" y2="14" stroke="#9a759a" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="8" y1="18" x2="20" y2="18" stroke="#9a759a" strokeWidth="1.5" strokeLinecap="round" />
          <line x1="11" y1="9" x2="11" y2="19" stroke="#b89ab8" strokeWidth="1" />
          <line x1="17" y1="9" x2="17" y2="19" stroke="#b89ab8" strokeWidth="1" />
        </svg>
        <span className={styles.logoText}>
          Off<span className={styles.logoAccent}>Rails</span>
        </span>
        <span className={styles.tagline}>agent trace anomaly detection</span>
      </header>

      <main className={styles.main}>
        {/* Panel 1 — Input */}
        <div className={styles.panel}>
          <PanelHeader active={!!traceInput} label="Agent Trace" step="01">
            {/* Sample buttons injected into the header row */}
            <div className={styles.sampleBtns}>
              <button className={styles.sampleBtn} onClick={() => loadSample('good')}>✓ normal</button>
              <button className={styles.sampleBtn} onClick={() => loadSample('bad')}>⚠ anomalous</button>
            </div>
          </PanelHeader>
          <div className={styles.panelBody}>
            <textarea
              className={styles.textarea}
              value={traceInput}
              onChange={e => setTraceInput(e.target.value)}
              placeholder={
                'Paste your agent trace JSON here.\n\nExpected format:\n{\n  "conversations": [\n    {"from": "user", "value": "..."},\n    {"from": "tool_call", "value": "..."},\n    {"from": "observation", "value": "..."},\n    {"from": "assistant", "value": "..."}\n  ]\n}'
              }
            />
            {traceInput && (
              <div className={styles.charCount}>{traceInput.length} chars</div>
            )}
            {error && <div className={styles.errorMsg}>{error}</div>}
          </div>
          <div className={styles.panelFooter}>
            <button
              className={styles.btnInvestigate}
              onClick={handleInvestigate}
              disabled={loading || !traceInput.trim()}
            >
              {loading ? (<><Spinner /><span>Investigating...</span></>) : 'Investigate'}
            </button>
          </div>
        </div>

        {/* Panel 2 — Flow */}
        <div className={styles.panel}>
          <PanelHeader active={!!steps} label="Trace Flow" step="02" />
          <div className={styles.panelBody}>
            {!steps && !loading && (
              <EmptyState
                icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="#b89ab8" strokeWidth="1" /><line x1="5" y1="8" x2="11" y2="8" stroke="#b89ab8" strokeWidth="1.2" strokeLinecap="round" /><line x1="8" y1="5" x2="8" y2="11" stroke="#b89ab8" strokeWidth="1.2" strokeLinecap="round" /></svg>}
                title="Trace visualization will appear here"
                sub="Each step, tool call, and observation rendered as a flow"
              />
            )}
            {loading && <div className={styles.loadingState}><Spinner /><span>Parsing trace...</span></div>}
            {steps && <FlowDiagram steps={steps} flags={result?.flags ?? []} />}
          </div>
          {steps && (
            <div className={styles.panelFooterMeta}>
              {steps.length} steps · {steps.filter(s => (s.from || '').toLowerCase() === 'tool_call').length} tool calls
            </div>
          )}
        </div>

        {/* Panel 3 — Output */}
        <div className={styles.panel}>
          <PanelHeader success={!!result} label="Analysis" step="03" />
          <div className={styles.panelBody}>
            {!result && !loading && (
              <EmptyState
                icon={<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><rect x="3" y="3" width="10" height="10" rx="2" stroke="#b89ab8" strokeWidth="1" /><line x1="6" y1="6" x2="10" y2="6" stroke="#b89ab8" strokeWidth="1.2" strokeLinecap="round" /><line x1="6" y1="8" x2="10" y2="8" stroke="#b89ab8" strokeWidth="1.2" strokeLinecap="round" /><line x1="6" y1="10" x2="8" y2="10" stroke="#b89ab8" strokeWidth="1.2" strokeLinecap="round" /></svg>}
                title="Model output will appear here"
                sub="Anomaly score, flags, and behavioral insights"
              />
            )}
            {loading && <div className={styles.loadingState}><Spinner /><span>Running analysis...</span></div>}
            {result && (
              <div className={styles.outputSection}>
                <ScoreRing score={result.score} />
                <div className={styles.divider} />
                <div className={styles.metricsGrid}>
                  {[
                    { val: result.metrics.steps,             lbl: 'total steps' },
                    { val: result.metrics.toolCalls,         lbl: 'tool calls' },
                    { val: result.metrics.uniqueTools,       lbl: 'unique tools' },
                    { val: `${result.metrics.repeatRatio}%`, lbl: 'repeat ratio' },
                  ].map(({ val, lbl }) => (
                    <div key={lbl} className={styles.metricCard}>
                      <div className={styles.metricVal}>{val}</div>
                      <div className={styles.metricLbl}>{lbl}</div>
                    </div>
                  ))}
                </div>
                {result.flags.length > 0 && (
                  <>
                    <div className={styles.divider} />
                    <div className={styles.flagsSection}>
                      <div className={styles.flagsTitle}>detected flags</div>
                      {result.flags.map((f, i) => (
                        <div key={i} className={styles.flagItem} style={{ animationDelay: `${i * 0.1}s` }}>
                          <div className={`${styles.flagDot} ${styles[`flag_${f.level}`]}`} />
                          <div>
                            <div className={styles.flagText}>{f.text}</div>
                            <div className={styles.flagTag}>{f.tag}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
                {result.flags.length === 0 && (
                  <div className={styles.noFlags}>No anomalies detected in this trace.</div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}