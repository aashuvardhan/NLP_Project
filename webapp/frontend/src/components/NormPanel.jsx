/**
 * NormPanel.jsx
 * Stage 1: Shows BERT norm classification result — single card, prominent verdict.
 */

export default function NormPanel({ isNorm, bertConfidence, threshold }) {
    const pct = Math.round((bertConfidence ?? 0) * 100)

    return (
        <div className="norm-panel animate-in">
            <div className="stage-label">🔍 Stage 1 — Norm Detection</div>

            <div className="bert-only-card">
                {/* Model badge */}
                <div className="bert-card-header">
                    <span className="model-name bert">BERT</span>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                        norm classifier · threshold {threshold.toFixed(2)}
                    </span>
                    <span className={`verdict-badge ${isNorm ? 'norm' : 'not-norm'}`} style={{ marginLeft: 'auto' }}>
                        {isNorm ? '✓ Norm' : '✗ Not a Norm'}
                    </span>
                </div>

                {/* Confidence bar */}
                <div className="confidence-bar-wrap" style={{ marginTop: '1rem' }}>
                    <div className="confidence-label">
                        <span>Norm confidence</span>
                        <span className="confidence-pct">{pct}%</span>
                    </div>
                    <div className="conf-track bert-track">
                        <div
                            className={`conf-fill ${isNorm ? 'norm-fill' : 'not-norm-fill'}`}
                            style={{ width: `${pct}%` }}
                        />
                        {/* Threshold marker */}
                        <div
                            className="threshold-marker"
                            style={{ left: `${threshold * 100}%` }}
                            title={`Threshold: ${threshold.toFixed(2)}`}
                        />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.25rem' }}>
                        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                            ▲ threshold at {Math.round(threshold * 100)}%
                        </span>
                    </div>
                </div>
            </div>

            {/* Verdict banner */}
            <div className={`majority-verdict ${isNorm ? 'is-norm' : 'not-norm'}`}>
                {isNorm
                    ? '✅ BERT detected a Cultural Norm — running country prediction across all models…'
                    : '❌ Not a cultural norm — no country prediction needed.'}
            </div>
        </div>
    )
}
