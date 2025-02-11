// Get canvas context
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set up canvas settings
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'black';

let drawing = false; // Track drawing state

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

// Clear canvas and reset prediction
document.getElementById('clear-btn').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = '';
    document.getElementById('prediction').style.color = '#000'; // Reset to black
    clearChart(); // Clear chart
});

// Handle prediction request
document.getElementById('predict-btn').addEventListener('click', async () => {
    const canvas = document.getElementById('canvas');
    const imageData = canvas.toDataURL('image/png');  // Get image data as base64 string

    // Process image to make empty space white
    const img = new Image();
    img.src = imageData;
    img.onload = () => {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = img.width;
        tempCanvas.height = img.height;
        tempCtx.drawImage(img, 0, 0);

        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const data = imageData.data;

        // Set all pixels close to black (background) to white
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];

            if (r < 50 && g < 50 && b < 50) {
                data[i] = 255;     // R
                data[i + 1] = 255; // G
                data[i + 2] = 255; // B
            }
        }

        tempCtx.putImageData(imageData, 0, 0);
        const processedImageData = tempCanvas.toDataURL('image/png');
        sendImageToBackend(processedImageData); // Send processed image to backend
    };
});

// Send image data to the backend
async function sendImageToBackend(imageData) {
    const formData = new FormData();
    formData.append('image', imageData.split(',')[1]); // Extract base64 string

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    document.getElementById('prediction').textContent = result.label; // Show prediction label
    document.getElementById('grayscale-img').src = `data:image/png;base64,${result.grayscale_image_base64}`;
    document.getElementById('processed-img').src = `data:image/png;base64,${result.processed_image_base64}`;

    document.getElementById('grayscale-img').classList.remove('d-none');
    document.getElementById('processed-img').classList.remove('d-none');

    updateChart(result.probabilities); // Update chart with probabilities
}

// Define bar colors
const barColors = [
    'rgba(0, 122, 255, 0.8)',  // Apple Blue
    'rgba(88, 86, 214, 0.8)',  // Deep Purple
    'rgba(255, 59, 48, 0.8)',  // Apple Red
    'rgba(255, 149, 0, 0.8)',  // Apple Orange
    'rgba(255, 204, 0, 0.8)',  // Apple Yellow
    'rgba(76, 217, 100, 0.8)', // Apple Green
    'rgba(90, 200, 250, 0.8)', // Light Blue
    'rgba(255, 45, 85, 0.8)',  // Dark Pink
    'rgba(174, 82, 255, 0.8)', // Light Purple
    'rgba(142, 142, 147, 0.8)' // Apple Gray
];

// Initialize Chart.js
const ctxChart = document.getElementById('bar-chart').getContext('2d');
const probabilityChart = new Chart(ctxChart, {
    type: 'bar',
    data: {
        labels: Array.from({ length: 10 }, (_, i) => i),
        datasets: [{
            label: 'Prediction Probabilities',
            data: Array(10).fill(0), // Initial empty data for probabilities
            backgroundColor: barColors,
            borderRadius: 10,  // Rounded bars
            borderWidth: 0,
            hoverBackgroundColor: 'rgba(255, 255, 255, 0.9)' // Light glow on hover
        }]
    },
    options: {
        responsive: true,
        animation: {
            duration: 1200,
            easing: 'easeOutQuart',
        },
        scales: {
            x: {
                title: { display: true, text: 'Digit Class (0-9)', font: { size: 14, weight: 'bold' } },
                ticks: { font: { size: 12 } }
            },
            y: {
                beginAtZero: true,
                max: 100, // Max set to 100 for percentage scale
                title: { display: true, text: 'Probability (%)', font: { size: 14, weight: 'bold' } },
                ticks: { font: { size: 12 }, stepSize: 10, callback: (value) => value + '%' },
                grid: { color: 'rgba(200, 200, 200, 0.2)' }
            }
        },
        plugins: {
            legend: { display: false }, // Hide legend
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.75)',
                titleFont: { size: 14, weight: 'bold' },
                bodyFont: { size: 12 },
                padding: 10,
                callbacks: {
                    label: (tooltipItem) => `Probability: ${tooltipItem.raw.toFixed(2)}%`
                }
            }
        }
    }
});

// Update chart and change prediction color
function updateChart(probabilities) {
    const percentageProbabilities = probabilities.map(p => p * 100);
    probabilityChart.data.datasets[0].data = percentageProbabilities;
    probabilityChart.update();

    const maxIndex = probabilities.indexOf(Math.max(...probabilities));
    document.getElementById('prediction').style.color = barColors[maxIndex]; // Match prediction color to highest probability
}

// Clear chart
function clearChart() {
    probabilityChart.data.datasets[0].data = Array(10).fill(0);
    probabilityChart.update();
}
