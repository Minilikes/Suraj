document.addEventListener("DOMContentLoaded", () => {
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
    const gaugeFill       = document.getElementById("g-fill");
    const statusDot       = document.getElementById("status-dot");
    const statusText      = document.getElementById("status-text");
    const apiStatus       = document.getElementById("api-status");
    const cityR2          = document.getElementById("city-r2");
    const cityClimate     = document.getElementById("city-climate");

    const GAUGE_RADIUS        = 80;
    const GAUGE_CIRCUMFERENCE = Math.PI * GAUGE_RADIUS;
    const MAX_IRRADIANCE      = 10.0;
    gaugeFill.style.strokeDasharray  = GAUGE_CIRCUMFERENCE;
    gaugeFill.style.strokeDashoffset = GAUGE_CIRCUMFERENCE;

    const CITY_DATA = {
        Jabalpur: { cloud:50, temp:25, humidity:60, yesterday:5.0, r2:"0.845", climate:"Tropical" },
        Bhopal:   { cloud:45, temp:26, humidity:55, yesterday:5.2, r2:"0.875", climate:"Tropical" },
        Delhi:    { cloud:35, temp:28, humidity:45, yesterday:5.5, r2:"0.885", climate:"Semi-Arid" },
        Mumbai:   { cloud:60, temp:30, humidity:75, yesterday:4.8, r2:"0.891", climate:"Coastal" },
        Jaipur:   { cloud:25, temp:32, humidity:35, yesterday:6.0, r2:"0.859", climate:"Arid" },
        Ladakh:   { cloud:20, temp:10, humidity:30, yesterday:6.5, r2:"0.830", climate:"Cold-Arid" },
    };

    function updateLabels() {
        cloudVal.textContent     = cloudSlider.value + "%";
        tempVal.textContent      = tempSlider.value + " \u00b0C";
        humidityVal.textContent  = humiditySlider.value + "%";
        yesterdayVal.textContent = parseFloat(yesterdaySlider.value).toFixed(1) + " MJ/m\u00b2";
    }

    function updateCityMeta() {
        const d = CITY_DATA[citySelect.value];
        if (cityR2)      cityR2.textContent      = d.r2;
        if (cityClimate) cityClimate.textContent = d.climate;
    }

    async function getPrediction() {
        const city      = citySelect.value;
        const cloud     = parseFloat(cloudSlider.value);
        const temp      = parseFloat(tempSlider.value);
        const humidity  = parseFloat(humiditySlider.value);
        const yesterday = parseFloat(yesterdaySlider.value);

        apiStatus.textContent = "Predicting\u2026";
        apiStatus.style.color = "#00e5ff";

        try {
            const res = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ city, cloud, temp, humidity, yesterday })
            });
            if (!res.ok) throw new Error();
            const data = await res.json();
            apiStatus.textContent = "";
            updateGauge(data.prediction);
            updateStatus(data.prediction);
        } catch {
            apiStatus.textContent = "Backend offline \u2014 run app.py";
            apiStatus.style.color = "#ff4444";
            predictionVal.textContent = "--";
        }
    }

    function updateGauge(value) {
        predictionVal.textContent = value.toFixed(2);
        const pct    = Math.min(value / MAX_IRRADIANCE, 1);
        const offset = GAUGE_CIRCUMFERENCE - pct * GAUGE_CIRCUMFERENCE;
        gaugeFill.style.strokeDashoffset = offset;

        let color, shadow;
        if (value >= 7)      { color = "#00e5ff"; shadow = "0 0 22px rgba(0,229,255,0.5)"; }
        else if (value >= 4) { color = "#ff9500"; shadow = "0 0 22px rgba(255,149,0,0.5)"; }
        else                 { color = "#ff4444"; shadow = "0 0 22px rgba(255,68,68,0.5)"; }

        gaugeFill.style.stroke         = color;
        predictionVal.style.textShadow = shadow;
        predictionVal.style.color      = "#fff";
    }

    function updateStatus(value) {
        statusDot.className = "dot";
        if (value >= 7) {
            statusDot.classList.add("c");
            statusText.textContent = "High output expected";
            statusText.style.color = "#00e5ff";
        } else if (value >= 4) {
            statusDot.classList.add("o");
            statusText.textContent = "Moderate output";
            statusText.style.color = "#ff9500";
        } else {
            statusDot.classList.add("r");
            statusText.textContent = "Low output \u2014 heavy coverage";
            statusText.style.color = "#ff4444";
        }
    }

    citySelect.addEventListener("change", () => {
        const d = CITY_DATA[citySelect.value];
        cloudSlider.value = d.cloud; tempSlider.value = d.temp;
        humiditySlider.value = d.humidity; yesterdaySlider.value = d.yesterday;
        updateLabels(); updateCityMeta(); getPrediction();
    });

    [cloudSlider, tempSlider, humiditySlider, yesterdaySlider].forEach(s =>
        s.addEventListener("input", () => { updateLabels(); getPrediction(); })
    );

    updateLabels(); updateCityMeta(); getPrediction();
});
