/**
 * CountryPanel.jsx
 * Stage 2: Comparative stats — top-3 country predictions from DeBERTa, BERT, and RoBERTa.
 * Highlights when all three models agree on the #1 prediction.
 */

const MODEL_LABELS = { bert: 'BERT', deberta: 'DeBERTa', roberta: 'RoBERTa' }
const MODEL_COLORS = { bert: 'bert', deberta: 'deberta', roberta: 'roberta' }
const MEDALS = ['🥇', '🥈', '🥉']
const MODEL_ORDER = ['bert', 'deberta', 'roberta']

function CountryModelColumn({ modelKey, predictions }) {
    return (
        <div className="country-model-card animate-in">
            <div className="country-model-header">
                <span className={`model-name ${MODEL_COLORS[modelKey]}`}>
                    {MODEL_LABELS[modelKey]}
                </span>
                <span style={{ fontSize: '0.70rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
                    country model
                </span>
            </div>

            <div className="country-predictions">
                {predictions.map((pred, idx) => {
                    const pct = Math.round(pred.confidence * 100)
                    return (
                        <div key={pred.country} className="country-entry">
                            <div className="country-row">
                                <span className="medal">{MEDALS[idx] || '•'}</span>
                                <span className="country-flag">{pred.flag}</span>
                                <span className="country-name">{pred.country}</span>
                                <span className="country-pct">{pct}%</span>
                            </div>
                            <div className="country-bar-wrap">
                                <div className="country-track">
                                    <div
                                        className={`country-fill rank-${idx}`}
                                        style={{ width: `${pct}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}

export default function CountryPanel({ countryResults }) {
    if (!countryResults) return null

    // Check consensus: all 3 available models agree on #1 country
    const topCountries = MODEL_ORDER
        .filter(k => countryResults[k]?.length > 0)
        .map(k => countryResults[k][0]?.country)
        .filter(Boolean)

    const allAgree = topCountries.length >= 2 && topCountries.every(c => c === topCountries[0])

    return (
        <div className="animate-in" style={{ marginTop: '1.5rem' }}>
            <div className="stage-label">🌍 Stage 2 — Country Prediction</div>
            <p className="stage-sublabel">
                Which country does each model think this norm belongs to?
            </p>

            {/* Consensus banner */}
            {allAgree && (
                <div className="consensus-banner">
                    <span className="consensus-flag">{countryResults[MODEL_ORDER.find(k => countryResults[k]?.length > 0)]?.[0]?.flag}</span>
                    <div>
                        <div className="consensus-title">All models agree!</div>
                        <div className="consensus-country">{topCountries[0]}</div>
                    </div>
                    <span className="consensus-badge">🤝 Consensus</span>
                </div>
            )}

            {/* Model columns */}
            <div className="country-grid">
                {MODEL_ORDER.map(key =>
                    countryResults[key]?.length > 0
                        ? <CountryModelColumn key={key} modelKey={key} predictions={countryResults[key]} />
                        : null
                )}
            </div>
        </div>
    )
}
