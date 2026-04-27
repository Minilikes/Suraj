js = r"""document.addEventListener("DOMContentLoaded", () => {
    const citySelect      = document.getElementById("city-select");
    const cloudSlider     = document.getElementById("cloud-fraction");
    const tempSlider      = document.getElementById("temperature");
    const humiditySlider  = document.getElementById("humidity");
    const yesterdaySlider = document.getElementById("yesterday-irradiance");
    const cloudVal        = document.getElementById("cloud-val");
    const tempVal         = document.getElementById("temp-val");
    const humidityVal     = document.getElementById("humidity-val");
    const yesterdayVal    = document.getElementById("yesterday-val");
    const predictionVal   = document.getElementById("prediction-val");
    const gaugeFill       = document.getElementById("gauge-fill");
    const statusDot       = document.getElementById("status-dot");
    const statusText      = document.getElementById("status-text");
    const apiStatus       = document.getElementById("api-status");

    const GAUGE_RADIUS        = 80;
    const GAUGE_CIRCUMFERENCE = Math.PI * GAUGE_RADIUS;
    const MAX_IRRADIANCE      = 10.0;
    gaugeFill.style.strokeDasharray = GAUGE_CIRCUMFERENCE;

    const CITY_DEFAULTS = {
        Jabalpur: { cloud: 50, temp: 25, humidity: 60, yesterday: 5.0 },
        Bhopal:   { cloud: 45, temp: 26, humidity: 55, yesterday: 5.2 },
        Delhi:    { cloud: 35, temp: 28, humidity: 45, yesterday: 5.5 },
        Mumbai:   { cloud: 60, temp: 30, humidity: 75, yesterday: 4.8 },
        Jaipur:   { cloud: 25, temp: 32, humidity: 35, yesterday: 6.0 },
        Ladakh:   { cloud: 20, temp: 10, humidity: 30, yesterday: 6.5 },
    };

    function updateLabels() {
        cloudVal.textContent     = cloudSlider.value + "%";
        tempVal.textContent      = tempSlider.value + " \u00b0C";
        humidityVal.textContent  = humiditySlider.value + "%";
        yesterdayVal.textContent = parseFloat(yesterdaySlider.value).toFixed(1) + " MJ/m\u00b2";
    }

    async function getPrediction() {
        const city      = citySelect.value;
        const cloud     = parseFloat(cloudSlider.value);
        const temp      = parseFloat(tempSlider.value);
        const humidity  = parseFloat(humiditySlider.value);
        const yesterday = parseFloat(yesterdaySlider.value);

        apiStatus.textContent = "Predicting...";
        apiStatus.style.color = "var(--neon-cyan)";

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ city, cloud, temp, humidity, yesterday })
            });
            if (!response.ok) throw new Error("Server error");
            const data = await response.json();
            apiStatus.textContent = "";
            updateGauge(data.prediction);
            updateStatus(data.prediction);
        } catch (err) {
            apiStatus.textContent = "Backend offline - run app.py first";
            apiStatus.style.color = "var(--neon-red)";
            predictionVal.textContent = "--";
        }
    }

    function updateGauge(value) {
        predictionVal.textContent = value.toFixed(2);
        const percentage = value / MAX_IRRADIANCE;
        const offset     = GAUGE_CIRCUMFERENCE - (percentage * GAUGE_CIRCUMFERENCE);
        gaugeFill.style.strokeDashoffset = offset;
        let color;
        if (value >= 7)      color = "var(--neon-cyan)";
        else if (value >= 4) color = "var(--neon-orange)";
        else                 color = "var(--neon-red)";
        gaugeFill.style.stroke = color;
        gaugeFill.style.filter = "drop-shadow(0 0 8px " + color + ")";
        predictionVal.style.color      = "#fff";
        predictionVal.style.textShadow = "0 0 15px " + color;
    }

    function updateStatus(value) {
        statusDot.className = "status-dot";
        if (value >= 7) {
            statusDot.classList.add("green");
            statusText.textContent = "High Output Expected";
            statusText.style.color = "var(--neon-cyan)";
        } else if (value >= 4) {
            statusDot.classList.add("yellow");
            statusText.textContent = "Moderate Output";
            statusText.style.color = "var(--neon-orange)";
        } else {
            statusDot.classList.add("red");
            statusText.textContent = "Low Output / Heavy Coverage";
            statusText.style.color = "var(--neon-red)";
        }
    }

    citySelect.addEventListener("change", () => {
        const d = CITY_DEFAULTS[citySelect.value];
        cloudSlider.value     = d.cloud;
        tempSlider.value      = d.temp;
        humiditySlider.value  = d.humidity;
        yesterdaySlider.value = d.yesterday;
        updateLabels();
        getPrediction();
    });

    [cloudSlider, tempSlider, humiditySlider, yesterdaySlider].forEach(s => {
        s.addEventListener("input", () => { updateLabels(); getPrediction(); });
    });

    updateLabels();
    getPrediction();
});
"""

with open("dashboard/script.js", "w", encoding="utf-8") as f:
    f.write(js)
print("script.js written OK")
