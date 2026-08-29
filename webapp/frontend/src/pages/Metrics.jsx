import { useState } from 'react'

// ── Hardcoded curated plots ───────────────────────────────────────────────────
const PHASE1_PLOTS = [
    {
        file: 'bert_training_curves.png',
        label: 'BERT — Training Curves',
        tag: 'Phase 1',
        base: '/plots/',
    },
    {
        file: 'bert_confusion_matrix.png',
        label: 'BERT — Confusion Matrix',
        tag: 'Phase 1',
        base: '/plots/',
    },
]

const PHASE2_PLOTS = [
    {
        file: 'country_bert_training_curves.png',
        label: 'BERT — Country Training Curves',
        tag: 'Phase 2',
        base: '/plots/country/',
    },
    {
        file: 'country_deberta_training_curves.png',
        label: 'DeBERTa — Country Training Curves',
        tag: 'Phase 2',
        base: '/plots/country/',
    },
    {
        file: 'country_roberta_training_curves.png',
        label: 'RoBERTa — Country Training Curves',
        tag: 'Phase 2',
        base: '/plots/country/',
    },
]

const MODEL_COLORS = { BERT: '#38bdf8', DeBERTa: '#a78bfa', RoBERTa: '#fb923c' }

function PlotCard({ file, label, tag, base, onExpand }) {
    const model = label.split(' — ')[0]
    const accent = MODEL_COLORS[model] || 'var(--accent)'
    const src = `${base}${file}`
    return (
        <div
            className="plot-card-new"
            onClick={() => onExpand(src)}
            style={{ '--card-accent': accent }}
        >
            <div className="plot-card-img-wrap">
                <img
                    src={src}
                    alt={label}
                    loading="lazy"
                    className="plot-card-img"
                />
                <div className="plot-card-zoom-hint">🔍 Click to expand</div>
            </div>
            <div className="plot-card-footer">
                <div className="plot-card-meta">
                    <span className="plot-model-badge" style={{ color: accent }}>{model}</span>
                    <span className="plot-phase-badge">{tag}</span>
                </div>
                <div className="plot-card-title">{label.split(' — ')[1]}</div>
            </div>
        </div>
    )
}

function PlotsSection({ title, subtitle, plots, onExpand }) {
    return (
        <div className="metrics-section">
            <div className="section-title">{title}</div>
            {subtitle && <p className="stage-sublabel" style={{ marginBottom: '1rem' }}>{subtitle}</p>}
            <div className="plots-grid-new">
                {plots.map(p => (
                    <PlotCard key={p.file} {...p} onExpand={onExpand} />
                ))}
            </div>
        </div>
    )
}

function Lightbox({ file, onClose }) {
    if (!file) return null
    return (
        <div className="lightbox-overlay" onClick={onClose}>
            <img src={file} alt="expanded plot" />
        </div>
    )
}

export default function Metrics() {
    const [lightbox, setLightbox] = useState(null)

    return (
        <div>
            <div className="page-hero">
                <h1>Model <span className="grad">Metrics</span></h1>
                <p>Training curves and confusion matrices for the norm and country classifiers.</p>
            </div>

            {/* Phase 1 */}
            <PlotsSection
                title="🔍 Phase 1 — Norm Classification (BERT)"
                subtitle="Training loss / F1 curves and test-set confusion matrix for the BERT norm classifier."
                plots={PHASE1_PLOTS}
                onExpand={setLightbox}
            />

            {/* Phase 2 */}
            <PlotsSection
                title="🌍 Phase 2 — Country Prediction"
                subtitle="Training curves for each country classifier backbone — compare how BERT, DeBERTa, and RoBERTa learned."
                plots={PHASE2_PLOTS}
                onExpand={setLightbox}
            />

            <Lightbox file={lightbox} onClose={() => setLightbox(null)} />
        </div>
    )
}
