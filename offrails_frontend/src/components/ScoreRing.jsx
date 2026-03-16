import React, { useState, useRef } from 'react';
import styles from './ScoreRing.module.css';

function InfoTooltip() {
  const [pos, setPos] = useState(null);
  const ref = useRef(null);

  const handleEnter = () => {
    if (ref.current) {
      const rect = ref.current.getBoundingClientRect();
      setPos({ top: rect.bottom + 8, left: rect.left + rect.width / 2 });
    }
  };

  return (
    <>
      <div
        ref={ref}
        onMouseEnter={handleEnter}
        onMouseLeave={() => setPos(null)}
        style={{
          width: 14, height: 14,
          borderRadius: '50%',
          border: '1px solid #c2b0ab',
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'default',
          color: '#9b8a86',
          fontSize: 9,
          fontWeight: 600,
          userSelect: 'none',
          flexShrink: 0,
        }}
      >
        i
      </div>
      {pos && (
        <div style={{
          position: 'fixed',
          top: pos.top,
          left: pos.left,
          transform: 'translateX(-50%)',
          width: 220,
          background: '#2a2420',
          color: '#e8dde8',
          fontSize: 11,
          lineHeight: 1.6,
          padding: '10px 12px',
          borderRadius: 8,
          boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
          fontFamily: '-apple-system, SF Pro Text, sans-serif',
          zIndex: 9999,
          pointerEvents: 'none',
          textAlign: 'left',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4, color: '#f0e8f0' }}>
            How the score is calculated
          </div>
          The anomaly score is the model's predicted probability (0–100%) that this trace is anomalous.
          Produced by an XGBoost classifier trained on 25+ features including tool repetition,
          error observations, linguistic signals, and trace structure.
        </div>
      )}
    </>
  );
}

export default function ScoreRing({ score }) {
  const r = 42;
  const circ = 2 * Math.PI * r;
  const fill = circ * (1 - score / 100);
  const color = score > 60 ? '#c0614e' : score > 30 ? '#c49040' : '#7a9e68';
  const verdict = score > 60 ? 'anomalous' : score > 30 ? 'warning' : 'normal';
  const label = score > 60 ? 'Anomalous' : score > 30 ? 'Review needed' : 'Normal';

  return (
    <div className={styles.wrap}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}>
        <span className={styles.labelTop}>anomaly score</span>
        <InfoTooltip />
      </div>
      <svg width="110" height="110" viewBox="0 0 110 110" className={styles.svg}>
        <circle cx="55" cy="55" r={r} fill="none" stroke="#ede5dc" strokeWidth="7" />
        <circle
          cx="55" cy="55" r={r}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeDasharray={circ}
          strokeDashoffset={fill}
          strokeLinecap="round"
          transform="rotate(-90 55 55)"
          style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.4,0,0.2,1)' }}
        />
        <text x="55" y="52" textAnchor="middle"
          style={{ fill: '#2c2422', fontSize: '26px', fontWeight: 600, fontFamily: '-apple-system, SF Pro Display, sans-serif' }}>
          {score}
        </text>
        <text x="55" y="66" textAnchor="middle"
          style={{ fill: '#9b8a86', fontSize: '10px', fontFamily: '-apple-system, SF Pro Text, sans-serif' }}>
          out of 100
        </text>
      </svg>
      <span className={`${styles.chip} ${styles[verdict]}`}>{label}</span>
    </div>
  );
}