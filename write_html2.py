html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S.U.R.A.J. Dashboard</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
<div class="dashboard-container">

    <header>
        <img src="assets/icon.png" alt="S.U.R.A.J." class="header-icon">
        <div class="header-text">
            <h1>S.U.R.A.J.</h1>
            <p>Solar Utility &amp; Radiance Analytical Judgment</p>
        </div>
        <div class="header-badges">
            <span class="badge badge-cyan">NASA POWER</span>
            <span class="badge badge-cyan">Random Forest</span>
            <span class="badge badge-live">Live</span>
        </div>
    </header>

    <div class="city-selector-wrapper">
        <label for="city-select">Location</label>
        <select id="city-select">
            <option value="Jabalpur">Jabalpur, Madhya Pradesh</option>
            <option value="Bhopal">Bhopal, Madhya Pradesh</option>
            <option value="Delhi">Delhi</option>
            <option value="Mumbai">Mumbai, Maharashtra</option>
            <option value="Jaipur">Jaipur, Rajasthan</option>
            <option value="Ladakh">Ladakh (UT)</option>
        </select>
        <div class="city-meta">
            <div class="city-stat">
                <div class="city-stat-label">Model R\u00b2</div>
                <div class="city-stat-value" id="city-r2">0.845</div>
            </div>
            <div class="city-stat">
                <div class="city-stat-label">Climate</div>
                <div class="city-stat-value" id="city-climate">Tropical</div>
            </div>
        </div>
    </div>

    <main class="dashboard-grid">

        <section class="panel controls-panel glass-panel">
            <h2>Environmental Inputs</h2>

            <div class="slider-group">
                <div class="slider-label-row">
                    <div class="slider-icon-name">
                        <div class="slider-icon">
                            <svg viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="1.5">
                                <circle cx="8" cy="8" r="3"/>
                                <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41"/>
                            </svg>
                        </div>
                        Cloud Fraction
                    </div>
                    <span id="cloud-val" class="value-display">50%</span>
                </div>
                <input type="range" id="cloud-fraction" min="0" max="100" value="50">
            </div>

            <div class="slider-group">
                <div class="slider-label-row">
                    <div class="slider-icon-name">
                        <div class="slider-icon">
                            <svg viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="1.5">
                                <path d="M8 2v7"/>
                                <circle cx="8" cy="12" r="2.5"/>
                                <path d="M11 5.5a4 4 0 0 1 0 5.66"/>
                            </svg>
                        </div>
                        Temperature
                    </div>
                    <span id="temp-val" class="value-display">25 \u00b0C</span>
                </div>
                <input type="range" id="temperature" min="10" max="45" value="25">
            </div>

            <div class="slider-group">
                <div class="slider-label-row">
                    <div class="slider-icon-name">
                        <div class="slider-icon">
                            <svg viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="1.5">
                                <path d="M8 2c0 0-5 4-5 8a5 5 0 0 0 10 0c0-4-5-8-5-8z"/>
                            </svg>
                        </div>
                        Relative Humidity
                    </div>
                    <span id="humidity-val" class="value-display">50%</span>
                </div>
                <input type="range" id="humidity" min="10" max="100" value="50">
            </div>

            <div class="slider-group">
                <div class="slider-label-row">
                    <div class="slider-icon-name">
                        <div class="slider-icon">
                            <svg viewBox="0 0 16 16" fill="none" stroke="#64748b" stroke-width="1.5">
                                <path d="M2 8h12M8 2l6 6-6 6"/>
                            </svg>
                        </div>
                        Yesterday's Irradiance
                    </div>
                    <span id="yesterday-val" class="value-display">5.0 MJ/m\u00b2</span>
                </div>
                <input type="range" id="yesterday-irradiance" min="0" max="10" step="0.1" value="5.0">
            </div>
        </section>

        <section class="panel output-panel glass-panel">
            <h2>Predicted Solar Irradiance</h2>

            <div class="gauge-container">
                <svg class="gauge-svg" viewBox="0 0 200 110">
                    <path class="gauge-track" d="M 15 95 A 80 80 0 0 1 185 95" fill="none"/>
                    <path id="gauge-fill" class="gauge-fill" d="M 15 95 A 80 80 0 0 1 185 95" fill="none"/>
                </svg>
                <div class="gauge-readout">
                    <span id="prediction-val" class="large-value">--</span>
                    <span class="unit">MJ/m\u00b2</span>
                </div>
            </div>

            <div class="status-indicator">
                <div id="status-dot" class="status-dot green"></div>
                <span id="status-text">Awaiting prediction</span>
            </div>

            <div id="api-status" class="api-status"></div>
        </section>

    </main>

    <footer class="dashboard-footer">
        S.U.R.A.J. \u00b7 NASA POWER API \u00b7 5-Year Training Data 2019\u20132023 \u00b7 Random Forest Regressor
    </footer>

</div>
<script src="script.js"></script>
</body>
</html>"""

with open("dashboard/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("index.html written OK")
