import { useState, useRef } from 'react'
import NormPanel from '../components/NormPanel.jsx'
import CountryPanel from '../components/CountryPanel.jsx'

const EXAMPLE_SENTENCES = [
    "In Japan, people bow when greeting someone as a sign of respect.",
    "In India, people touch the feet of elders as a sign of respect.",
    "The sky appears blue due to Rayleigh scattering of sunlight.",
    "In Brazil, people greet friends with a kiss on the cheek.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "In North Korea, it is polite to use two hands when giving or receiving something.",
]

export default function Classifier() {
    const [sentence, setSentence] = useState('')
    const [threshold, setThreshold] = useState(0.6)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const textareaRef = useRef(null)

    async function handlePredict() {
        const text = sentence.trim()
        if (!text) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence: text, norm_threshold: threshold }),
            })
            if (!res.ok) throw new Error(`Server error: ${res.status}`)
            const data = await res.json()
            setResult(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }

    function handleKeyDown(e) {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handlePredict()
    }

    function loadExample(s) {
        setSentence(s)
        setResult(null)
        setError(null)
        textareaRef.current?.focus()
    }

    return (
        <div>
            {/* Hero */}
            <div className="page-hero">
                <h1>
                    <span className="grad">Cultural Norm</span> Classifier
                </h1>
                <p>
                    <strong>BERT</strong> decides whether your sentence is a cultural norm.
                    If it is, <strong>all three transformer models</strong> (BERT · DeBERTa · RoBERTa)
                    predict the country — so you can compare their answers side-by-side.
                </p>
            </div>

            {/* Input card */}
            <div className="card input-card">
                <div className="input-row">
                    <textarea
                        ref={textareaRef}
                        className="sentence-input"
                        placeholder="Type a sentence… (Ctrl+Enter to predict)"
                        value={sentence}
                        onChange={e => setSentence(e.target.value)}
                        onKeyDown={handleKeyDown}
                        id="sentence-input"
                    />
                    <button
                        id="predict-btn"
                        className="predict-btn"
                        onClick={handlePredict}
                        disabled={loading || !sentence.trim()}
                    >
                        {loading ? <><span className="spinner" />Predicting…</> : '⚡ Predict'}
                    </button>
                </div>

                {/* Threshold slider */}
                <div className="threshold-row">
                    <span>Norm threshold:</span>
                    <input
                        type="range" min="0.3" max="0.9" step="0.05"
                        value={threshold}
                        onChange={e => setThreshold(parseFloat(e.target.value))}
                    />
                    <span className="threshold-val">{threshold.toFixed(2)}</span>
                    <span style={{ marginLeft: 'auto', fontSize: '0.75rem' }}>Ctrl+Enter to predict</span>
                </div>

                {/* Example sentences */}
                <div style={{ marginTop: '0.85rem', display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', alignSelf: 'center' }}>Try:</span>
                    {EXAMPLE_SENTENCES.map(s => (
                        <button
                            key={s}
                            onClick={() => loadExample(s)}
                            style={{
                                fontSize: '0.72rem', padding: '0.25rem 0.65rem',
                                background: 'rgba(255,255,255,0.04)', border: '1px solid var(--border)',
                                borderRadius: '20px', color: 'var(--text-muted)', cursor: 'pointer',
                                transition: 'all 0.15s'
                            }}
                            onMouseOver={e => e.currentTarget.style.color = 'var(--text)'}
                            onMouseOut={e => e.currentTarget.style.color = 'var(--text-muted)'}
                        >
                            {s.length > 45 ? s.slice(0, 45) + '…' : s}
                        </button>
                    ))}
                </div>
            </div>

            {/* Error */}
            {error && <div className="error-msg">⚠️ {error}</div>}

            {/* Results */}
            {result && (
                <div className="card" style={{ marginTop: '0' }}>
                    {/* Sentence echo */}
                    <div style={{
                        fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1.25rem',
                        padding: '0.7rem 1rem', background: 'rgba(255,255,255,0.03)',
                        borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--accent)'
                    }}>
                        <em>"{result.sentence}"</em>
                    </div>

                    {/* Stage 1: BERT norm detection */}
                    <NormPanel
                        isNorm={result.is_norm}
                        bertConfidence={result.bert_norm_confidence}
                        threshold={threshold}
                    />

                    {/* Stage 2: country comparison (only if norm) */}
                    {result.is_norm && (
                        <CountryPanel countryResults={result.country_results} />
                    )}
                </div>
            )}
        </div>
    )
}
