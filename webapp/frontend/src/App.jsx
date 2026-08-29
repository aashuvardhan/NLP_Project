import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import Classifier from './pages/Classifier.jsx'
import Metrics from './pages/Metrics.jsx'
import About from './pages/About.jsx'

export default function App() {
    return (
        <BrowserRouter>
            <div className="app-shell">
                <nav className="nav">
                    <NavLink to="/" className="nav-logo">
                        🧭 <span>P3 NormClassifier</span>
                    </NavLink>
                    <div className="nav-links">
                        <NavLink to="/" end className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Classifier</NavLink>
                        <NavLink to="/metrics" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>Model Metrics</NavLink>
                        <NavLink to="/about" className={({ isActive }) => 'nav-link' + (isActive ? ' active' : '')}>About</NavLink>
                    </div>
                    <span className="nav-badge">DeBERTa · BERT · RoBERTa</span>
                </nav>

                <main className="main-content">
                    <Routes>
                        <Route path="/" element={<Classifier />} />
                        <Route path="/metrics" element={<Metrics />} />
                        <Route path="/about" element={<About />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    )
}
