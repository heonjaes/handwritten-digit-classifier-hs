// Ensure the canvas is cleared after a prediction
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set up canvas settings
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'black';

// Variables to hold canvas image data
let drawing = false;

// Start drawing when mouse is pressed
canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

// Draw on canvas when mouse is moved
canvas.addEventListener('mousemove', (e) => {
    if (drawing) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

// Stop drawing when mouse is released
canvas.addEventListener('mouseup', () => {
    drawing = false;
});

// Clear canvas
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = 'None';
    clearChart();
});

// Handle prediction request
document.getElementById('predict-btn').addEventListener('click', async () => {
    const imageData = canvas.toDataURL('image/png');  // Get image data as base64 string

    // Process image to make empty space white
    const img = new Image();
    img.src = imageData;
    img.onload = () => {
        // Create a canvas to process the image
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = img.width;
        tempCanvas.height = img.height;
        tempCtx.drawImage(img, 0, 0);

        // Get pixel data
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;

        // Set all pixels that are close to black (empty space) to white
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            // If the pixel is close to black (background), make it white
            if (r < 50 && g < 50 && b < 50) {
                data[i] = 255;     // R
                data[i + 1] = 255; // G
                data[i + 2] = 255; // B
            }
        }

        // Put the modified image back into the canvas
        tempCtx.putImageData(imageData, 0, 0);

        // Send the processed image to the backend as base64
        const processedImageData = tempCanvas.toDataURL('image/png');
        sendImageToBackend(processedImageData);
    };
});

// Send image data to the backend
async function sendImageToBackend(imageData) {
    const formData = new FormData();
    formData.append('image', imageData.split(',')[1]); // Extract base64 string without the prefix

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('prediction').textContent = result.label;

    // Update the chart with the probabilities
    updateChart(result.probabilities);
}

// Initialize Chart.js
const ctxChart = document.getElementById('bar-chart').getContext('2d');
const probabilityChart = new Chart(ctxChart, {
    type: 'bar',
    data: {
        labels: Array.from({ length: 10 }, (_, i) => i), // Labels for 10 classes
        datasets: [{
            label: 'Prediction Probabilities',
            data: Array(10).fill(0), // Initial empty data for probabilities
            backgroundColor: '#a8dadc',
            borderColor: '#a8dadc',
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Class Labels', // Label for the X-axis
                    color: '#333',
                    font: {
                        size: 14,
                    }
                }
            },
            y: {
                beginAtZero: true,
                max: 1,
                title: {
                    display: true,
                    text: 'Probability', // Label for the Y-axis
                    color: '#333',
                    font: {
                        size: 14,
                    }
                }
            }
        }
    }
});


// Update chart with new probabilities
function updateChart(probabilities) {
    probabilityChart.data.datasets[0].data = probabilities;
    probabilityChart.update();
}

// Clear chart
function clearChart() {
    probabilityChart.data.datasets[0].data = Array(10).fill(0);
    probabilityChart.update();
}
