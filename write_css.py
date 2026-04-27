css = """@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #080c14;
    --bg2: #0d1320;
    --bg3: #111827;
    --panel: rgba(255,255,255,0.04);
    --panel-border: rgba(255,255,255,0.08);
    --cyan: #00e5ff;
    --cyan-dim: rgba(0,229,255,0.15);
    --cyan-glow: rgba(0,229,255,0.4);
    --orange: #ff9500;
    --red: #ff4444;
    --text: #f0f6ff;
    --muted: #64748b;
    --font-head: 'Orbitron', sans-serif;
    --font-body: 'Inter', sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 2rem 1rem;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(0,229,255,0.05) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(0,80,120,0.08) 0%, transparent 60%);
}

.dashboard-container {
    width: 100%;
    max-width: 1080px;
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

/* ── HEADER ─────────────────────────────────────────────── */
header {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 2rem;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    position: relative;
    overflow: hidden;
}

header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    opacity: 0.6;
}

.header-icon {
    width: 52px;
    height: 52px;
    animation: spin-slow 12s linear infinite;
    filter: drop-shadow(0 0 12px var(--cyan));
    flex-shrink: 0;
}

@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.header-text { flex: 1; }

.header-text h1 {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 900;
    letter-spacing: 0.4rem;
    color: #fff;
    text-shadow: 0 0 20px var(--cyan-glow);
    line-height: 1;
}

.header-text p {
    color: var(--muted);
    font-size: 0.8rem;
    letter-spacing: 0.2rem;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.header-badges {
    display: flex;
    gap: 0.5rem;
    flex-shrink: 0;
}

.badge {
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: var(--font-head);
    font-weight: 600;
    letter-spacing: 0.05rem;
    border: 1px solid;
}

.badge-cyan {
    color: var(--cyan);
    border-color: rgba(0,229,255,0.3);
    background: rgba(0,229,255,0.07);
}

.badge-live {
    color: #4ade80;
    border-color: rgba(74,222,128,0.3);
    background: rgba(74,222,128,0.07);
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.badge-live::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 6px #4ade80;
    animation: blink 2s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── CITY SELECTOR ───────────────────────────────────────── */
.city-selector-wrapper {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.5rem;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
}

.city-selector-wrapper label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15rem;
    color: var(--muted);
    font-family: var(--font-head);
    white-space: nowrap;
}

.city-selector-wrapper select {
    flex: 1;
    background: var(--bg3);
    color: var(--text);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    font-family: var(--font-body);
    font-size: 0.95rem;
    cursor: pointer;
    outline: none;
    transition: border-color 0.2s;
    max-width: 320px;
}

.city-selector-wrapper select:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px var(--cyan-dim);
}

.city-meta {
    margin-left: auto;
    display: flex;
    gap: 1.5rem;
}

.city-stat {
    text-align: right;
}

.city-stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1rem;
}

.city-stat-value {
    font-family: var(--font-head);
    font-size: 0.85rem;
    color: var(--cyan);
    margin-top: 0.1rem;
}

/* ── GRID ────────────────────────────────────────────────── */
.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
}

@media (max-width: 750px) {
    .dashboard-grid { grid-template-columns: 1fr; }
    header { flex-wrap: wrap; }
    .header-badges { display: none; }
    .city-meta { display: none; }
}

/* ── PANELS ──────────────────────────────────────────────── */
.glass-panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 20px;
    padding: 1.75rem;
    transition: border-color 0.3s;
}

.glass-panel:hover {
    border-color: rgba(0,229,255,0.2);
}

h2 {
    font-family: var(--font-head);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.2rem;
    text-transform: uppercase;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

h2::before {
    content: '';
    display: block;
    width: 3px; height: 14px;
    background: var(--cyan);
    border-radius: 2px;
    box-shadow: 0 0 8px var(--cyan);
}

/* ── SLIDERS ─────────────────────────────────────────────── */
.slider-group {
    margin-bottom: 1.6rem;
}

.slider-group:last-child { margin-bottom: 0; }

.slider-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}

.slider-icon-name {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.9rem;
    color: var(--muted);
}

.slider-icon {
    width: 28px; height: 28px;
    border-radius: 8px;
    background: var(--bg3);
    border: 1px solid var(--panel-border);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.slider-icon svg { width: 14px; height: 14px; }

.value-display {
    font-family: var(--font-head);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--cyan);
    text-shadow: 0 0 8px var(--cyan-glow);
    min-width: 80px;
    text-align: right;
}

input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    background: transparent;
    cursor: pointer;
}

input[type=range]::-webkit-slider-runnable-track {
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
}

input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--bg);
    border: 2px solid var(--cyan);
    margin-top: -7.5px;
    box-shadow: 0 0 10px var(--cyan-glow);
    transition: transform 0.15s, box-shadow 0.15s;
}

input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.3);
    box-shadow: 0 0 16px var(--cyan);
}

/* ── OUTPUT PANEL ────────────────────────────────────────── */
.output-panel {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
}

.gauge-container {
    position: relative;
    width: 100%;
    max-width: 280px;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 0.5rem 0;
}

.gauge-svg { width: 100%; overflow: visible; }

.gauge-track {
    stroke: rgba(255,255,255,0.06);
    stroke-width: 12;
    stroke-linecap: round;
}

.gauge-fill {
    stroke: var(--cyan);
    stroke-width: 12;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.7s cubic-bezier(0.34,1.56,0.64,1), stroke 0.4s;
}

.gauge-readout {
    position: absolute;
    bottom: -10px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.large-value {
    font-family: var(--font-head);
    font-size: 3.2rem;
    font-weight: 900;
    color: #fff;
    line-height: 1;
    transition: color 0.4s, text-shadow 0.4s;
    text-shadow: 0 0 20px var(--cyan-glow);
}

.unit {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.2rem;
    letter-spacing: 0.05rem;
}

/* ── STATUS ──────────────────────────────────────────────── */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 1.2rem;
    background: var(--bg3);
    border-radius: 30px;
    border: 1px solid var(--panel-border);
    margin-top: 0.5rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.status-dot.green  { background: var(--cyan);   box-shadow: 0 0 8px var(--cyan);   animation: blink 2s infinite; }
.status-dot.yellow { background: var(--orange);  box-shadow: 0 0 8px var(--orange); animation: blink 2s infinite; }
.status-dot.red    { background: var(--red);     box-shadow: 0 0 8px var(--red);    animation: blink 2s infinite; }

#status-text {
    font-family: var(--font-head);
    font-size: 0.7rem;
    letter-spacing: 0.1rem;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── API STATUS ──────────────────────────────────────────── */
.api-status {
    font-size: 0.75rem;
    letter-spacing: 0.05rem;
    min-height: 1rem;
    text-align: center;
}

/* ── FOOTER ──────────────────────────────────────────────── */
.dashboard-footer {
    text-align: center;
    padding: 1rem;
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.1rem;
    text-transform: uppercase;
    border-top: 1px solid var(--panel-border);
}
"""

with open("dashboard/styles.css", "w", encoding="utf-8") as f:
    f.write(css)
print("styles.css written OK")
