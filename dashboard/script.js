// Wait for the entire HTML document to finish loading before running this script.
// This ensures that all the HTML elements exist before we try to manipulate them.
document.addEventListener('DOMContentLoaded', () => {

    // ---------------------------------------------------------
    // 1. GRAB ALL THE HTML ELEMENTS WE NEED
    // ---------------------------------------------------------
    
    // We get references to the four slider inputs using their IDs.
    // We will listen to these sliders to know when the user changes a value.
    const cloudSlider = document.getElementById('cloud-fraction');
    const tempSlider = document.getElementById('temperature');
    const humiditySlider = document.getElementById('humidity');
    const yesterdaySlider = document.getElementById('yesterday-irradiance');

    // We get references to the small text labels next to the sliders.
    // We will update these labels so the user can see the exact number they selected.
    const cloudVal = document.getElementById('cloud-val');
    const tempVal = document.getElementById('temp-val');
    const humidityVal = document.getElementById('humidity-val');
    const yesterdayVal = document.getElementById('yesterday-val');
    
    // We get references to the large gauge display elements.
    // 'predictionVal' is the big number showing the final calculation.
    const predictionVal = document.getElementById('prediction-val');
    // 'gaugeFill' is the glowing arc of the gauge that fills up.
    const gaugeFill = document.getElementById('gauge-fill');
    
    // We get references to the status indicator at the bottom (the dot and the text).
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');

    // ---------------------------------------------------------
    // 2. SETUP THE GAUGE MATH
    // ---------------------------------------------------------
    
    // The gauge is a half-circle drawn using an SVG (Scalable Vector Graphic).
    // The radius of our drawn circle is 80 units.
    const GAUGE_RADIUS = 80;
    // The circumference of a full circle is 2 * PI * Radius.
    // Since our gauge is exactly a half-circle, the total length of the line is just PI * Radius.
    const GAUGE_CIRCUMFERENCE = Math.PI * GAUGE_RADIUS; 
    
    // We define the absolute maximum solar irradiance value possible in our simulation (30 MJ/m²).
    const MAX_IRRADIANCE = 30.0;

    // Initially, we set the 'dasharray' (the length of the drawn line) to the full circumference.
    // We will then use 'dashoffset' later to "hide" part of the line depending on the value.
    gaugeFill.style.strokeDasharray = GAUGE_CIRCUMFERENCE;
    
    // ---------------------------------------------------------
    // 3. THE MAIN SIMULATION LOGIC FUNCTION
    // ---------------------------------------------------------
    
    // This function calculates the predicted solar energy whenever a slider is moved.
    function updateDashboard() {
        
        // Step A: Read the current numerical values from all four sliders.
        // 'parseFloat' converts the text value of the slider into a mathematical decimal number.
        const cloud = parseFloat(cloudSlider.value);
        const temp = parseFloat(tempSlider.value);
        const humidity = parseFloat(humiditySlider.value);
        const yesterday = parseFloat(yesterdaySlider.value);

        // Step B: Update the text labels on the screen so the user sees their chosen values.
        // The `${...}` syntax injects the number into a text string.
        cloudVal.textContent = `${cloud}%`;
        tempVal.textContent = `${temp} °C`;
        humidityVal.textContent = `${humidity}%`;
        yesterdayVal.textContent = `${yesterday.toFixed(1)} MJ/m²`; // toFixed(1) forces exactly 1 decimal place.

        // Step C: Perform the Machine Learning Simulation Math!
        
        // 1. Establish a 'Base' value. We assume tomorrow's weather is heavily influenced by yesterday's.
        // We take 60% of yesterday's value and add a baseline of 12.0 to smooth things out.
        let base = (yesterday * 0.6) + 12.0; 
        
        // 2. Calculate the 'Cloud Penalty'.
        // Clouds are the biggest blocker of solar energy. We use a math power function (Math.pow) 
        // to say that as clouds increase, the energy drops off drastically.
        let cloudMultiplier = 1.0 - (Math.pow(cloud / 100, 1.5) * 0.85);

        // 3. Calculate the 'Temperature Penalty'.
        // Solar panels work best at a cool 25°C. If it gets too hot or too cold, they lose efficiency.
        // Math.abs finds the absolute difference from 25.
        let tempPenalty = 1.0 - (Math.abs(temp - 25) * 0.004);

        // 4. Calculate the 'Humidity Penalty'.
        // High humidity means lots of water vapor in the air, which slightly scatters the sunlight.
        let humidityPenalty = 1.0 - ((humidity / 100) * 0.1);

        // 5. Calculate the Final Prediction!
        // We multiply the base value by all our penalty multipliers.
        let predicted = base * cloudMultiplier * tempPenalty * humidityPenalty;
        
        // 6. Enforce hard limits (Clamping).
        // It's impossible to have less than 0 or more than the MAX_IRRADIANCE, so we strictly enforce those boundaries.
        predicted = Math.max(0, Math.min(MAX_IRRADIANCE, predicted));

        // Step D: Send the final predicted number to our visual update functions.
        updateGauge(predicted);
        updateStatus(predicted);
    }

    // ---------------------------------------------------------
    // 4. THE GAUGE ANIMATION FUNCTION
    // ---------------------------------------------------------
    
    // This function takes the final number and physically animates the circular gauge.
    function updateGauge(value) {
        
        // Update the giant number text in the center of the gauge (rounded to 1 decimal).
        predictionVal.textContent = value.toFixed(1);

        // Calculate how "full" the gauge should be as a percentage (e.g., 15 out of 30 is 50% or 0.5).
        const percentage = value / MAX_IRRADIANCE;
        
        // Calculate the 'offset'. This is how much of the line we hide.
        // If percentage is 1 (100%), offset is 0 (hide nothing). If percentage is 0, offset is full (hide everything).
        const offset = GAUGE_CIRCUMFERENCE - (percentage * GAUGE_CIRCUMFERENCE);
        gaugeFill.style.strokeDashoffset = offset; // Apply the calculated offset to the CSS.

        // Determine the colors based on how good the output is.
        let color, shadow;
        if (value >= 20) {
            // High output: Glowing Cyan
            color = 'var(--neon-cyan)';
            shadow = '0 0 15px var(--neon-cyan)';
        } else if (value >= 10) {
            // Moderate output: Glowing Orange
            color = 'var(--neon-orange)';
            shadow = '0 0 15px var(--neon-orange)';
        } else {
            // Low output: Glowing Red
            color = 'var(--neon-red)';
            shadow = '0 0 15px var(--neon-red)';
        }

        // Apply the chosen colors and shadows to the gauge and text.
        gaugeFill.style.stroke = color;
        gaugeFill.style.filter = `drop-shadow(0 0 8px ${color})`;
        predictionVal.style.color = '#fff';
        predictionVal.style.textShadow = shadow;
    }

    // ---------------------------------------------------------
    // 5. THE STATUS INDICATOR FUNCTION
    // ---------------------------------------------------------
    
    // This function updates the small text and colored dot at the bottom of the screen.
    function updateStatus(value) {
        // Reset the dot to have no specific color classes.
        statusDot.className = 'status-dot';
        
        // Apply the correct color class and text description based on the value.
        if (value >= 20) {
            statusDot.classList.add('green'); // Actually applies cyan styling in CSS
            statusText.textContent = 'High Output Expected';
            statusText.style.color = 'var(--neon-cyan)';
        } else if (value >= 10) {
            statusDot.classList.add('yellow'); // Actually applies orange styling in CSS
            statusText.textContent = 'Moderate Output';
            statusText.style.color = 'var(--neon-orange)';
        } else {
            statusDot.classList.add('red');
            statusText.textContent = 'Low Output / Heavy Coverage';
            statusText.style.color = 'var(--neon-red)';
        }
    }

    // ---------------------------------------------------------
    // 6. INITIALIZATION (STARTING THE ENGINE)
    // ---------------------------------------------------------
    
    // We attach "Event Listeners" to the sliders. 
    // This tells the browser: "Whenever the user moves this input, run the 'updateDashboard' function immediately."
    cloudSlider.addEventListener('input', updateDashboard);
    tempSlider.addEventListener('input', updateDashboard);
    humiditySlider.addEventListener('input', updateDashboard);
    yesterdaySlider.addEventListener('input', updateDashboard);

    // Finally, we run the calculation once manually right when the page loads, 
    // so the gauge isn't empty before the user touches a slider.
    updateDashboard();
});
