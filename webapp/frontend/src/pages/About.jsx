export default function About() {
    return (
        <div>
            <div className="page-hero">
                <h1>About <span className="grad">P3</span></h1>
                <p>A transformer-based cultural norm classifier. BERT detects norms; DeBERTa, BERT, and RoBERTa compete on country prediction.</p>
            </div>

            {/* Pipeline */}
            <div className="card" style={{ marginBottom: '1.25rem' }}>
                <div className="section-title" style={{ marginBottom: '1rem' }}>🔀 Two-Stage Pipeline</div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                    {[
                        { icon: '📝', title: 'Input Sentence', desc: 'Any free-text sentence in English' },
                        { icon: '🔍', title: 'Stage 1 — Norm Detection (BERT)', desc: 'BERT alone decides: is this a cultural behavioural norm?' },
                        { icon: '❌', title: 'Not a Norm', desc: 'Sentence is factual/observational — pipeline stops here' },
                        { icon: '🌍', title: 'Stage 2 — Country Prediction', desc: 'DeBERTa · BERT · RoBERTa each predict top-3 countries — compare them side-by-side' },
                    ].map(step => (
                        <div key={step.title} className="pipeline-step">
                            <span className="pipeline-icon">{step.icon}</span>
                            <div className="pipeline-text">
                                <strong>{step.title}</strong>
                                {step.desc}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Stats */}
            <div className="about-grid" style={{ marginBottom: '1.25rem' }}>
                {[
                    { stat: '36,062', label: 'Total training sentences' },
                    { stat: '56', label: 'Countries in dataset' },
                    { stat: '~82%', label: 'Norm classifier accuracy' },
                    { stat: '~91%', label: 'Country classifier accuracy (DeBERTa)' },
                ].map(s => (
                    <div key={s.label} className="card" style={{ textAlign: 'center', padding: '1.25rem' }}>
                        <div className="stat-highlight">{s.stat}</div>
                        <div className="stat-label">{s.label}</div>
                    </div>
                ))}
            </div>

            {/* Models */}
            <div className="card">
                <div className="section-title" style={{ marginBottom: '0.85rem' }}>🤖 Models Used</div>
                <table className="metrics-table">
                    <thead>
                        <tr><th>Key</th><th>Checkpoint</th><th>Norm Acc.</th><th>Country Acc.</th></tr>
                    </thead>
                    <tbody>
                        <tr><td style={{ color: '#38bdf8', fontWeight: 700 }}>BERT</td><td>bert-base-uncased</td><td className="best-val">✓ Used</td><td>88.3%</td></tr>
                        <tr><td style={{ color: '#a78bfa', fontWeight: 700 }}>DeBERTa</td><td>microsoft/deberta-v3-base</td><td style={{ color: 'var(--text-muted)' }}>—</td><td>91.0%</td></tr>
                        <tr><td style={{ color: '#fb923c', fontWeight: 700 }}>RoBERTa</td><td>roberta-base</td><td style={{ color: 'var(--text-muted)' }}>—</td><td>87.5%</td></tr>
                    </tbody>
                </table>
            </div>

            {/* Data sources */}
            <div className="card" style={{ marginTop: '1.25rem' }}>
                <div className="section-title" style={{ marginBottom: '0.85rem' }}>📚 Data Sources</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.65rem' }}>
                    {[
                        { src: 'CultureBank Reddit', desc: '~14K norm sentences with cultural group labels' },
                        { src: 'CultureBank TikTok', desc: 'Norms sourced from TikTok caption analysis' },
                        { src: 'Wikipedia (SimpleWiki)', desc: '~20K factual sentences as non-norm negatives' },
                        { src: 'NormAD / CultureAtlas', desc: 'Additional country-labelled norm data for Phase 2' },
                    ].map(d => (
                        <div key={d.src} style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
                            <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.2rem' }}>{d.src}</div>
                            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{d.desc}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
