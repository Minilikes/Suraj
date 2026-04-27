html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S.U.R.A.J. Dashboard</title>
    <link rel="stylesheet" href="styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
</head>
<body>
    <div class="dashboard-container">

        <header>
            <img src="assets/icon.png" alt="S.U.R.A.J. Energy Icon" class="header-icon">
            <h1 class="glow-text">S.U.R.A.J.</h1>
            <p>Solar Energy Forecasting Dashboard</p>
        </header>

        <div class="city-selector-wrapper">
            <label for="city-select">Select City</label>
            <select id="city-select">
                <option value="Jabalpur">Jabalpur, Madhya Pradesh</option>
                <option value="Bhopal">Bhopal, Madhya Pradesh</option>
                <option value="Delhi">Delhi</option>
                <option value="Mumbai">Mumbai, Maharashtra</option>
                <option value="Jaipur">Jaipur, Rajasthan</option>
                <option value="Ladakh">Ladakh (UT)</option>
            </select>
        </div>

        <main class="dashboard-grid">

            <section class="panel controls-panel glass-panel">
                <h2>Environmental Inputs</h2>

                <div class="slider-group">
                    <label for="cloud-fraction">
                        <span>Cloud Fraction</span>
                        <span id="cloud-val" class="value-display">50%</span>
                    </label>
                    <input type="range" id="cloud-fraction" min="0" max="100" value="50">
                </div>

                <div class="slider-group">
                    <label for="temperature">
                        <span>Temperature</span>
                        <span id="temp-val" class="value-display">25 \u00b0C</span>
                    </label>
                    <input type="range" id="temperature" min="10" max="45" value="25">
                </div>

                <div class="slider-group">
                    <label for="humidity">
                        <span>Relative Humidity</span>
                        <span id="humidity-val" class="value-display">50%</span>
                    </label>
                    <input type="range" id="humidity" min="10" max="100" value="50">
                </div>

                <div class="slider-group">
                    <label for="yesterday-irradiance">
                        <span>Yesterday's Irradiance</span>
                        <span id="yesterday-val" class="value-display">5.0 MJ/m\u00b2</span>
                    </label>
                    <input type="range" id="yesterday-irradiance" min="0" max="10" step="0.1" value="5.0">
                </div>
            </section>

            <section class="panel output-panel glass-panel">
                <h2>Predicted Solar Irradiance</h2>

                <div class="gauge-container">
                    <svg class="gauge-svg" viewBox="0 0 200 100">
                        <path class="gauge-track" d="M 10 90 A 80 80 0 0 1 190 90" fill="none" />
                        <path id="gauge-fill" class="gauge-fill" d="M 10 90 A 80 80 0 0 1 190 90" fill="none" />
                    </svg>
                    <div class="gauge-readout">
                        <span id="prediction-val" class="large-value">--</span>
                        <span class="unit">MJ/m\u00b2</span>
                    </div>
                </div>

                <div class="status-indicator">
                    <div id="status-dot" class="status-dot green"></div>
                    <span id="status-text">Select conditions to predict</span>
                </div>

                <div id="api-status" class="api-status"></div>
            </section>

        </main>
    </div>

    <script src="script.js"></script>
</body>
</html>"""

with open("dashboard/index.html", "w", encoding="utf-8") as f:
    f.write(html)
print("index.html written OK")
