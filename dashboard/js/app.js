// Pest Detection Dashboard JavaScript

const API_BASE = 'http://localhost:8000/api';
let currentDevice = null;
let sensorChart = null;
let ndviChart = null;
let ws = null;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadDevices();
    setupEventListeners();
    connectWebSocket();
});

// Load available devices
async function loadDevices() {
    try {
        const response = await fetch(`${API_BASE}/sensors/all`);
        const devices = await response.json();
        
        const select = document.getElementById('deviceSelect');
        select.innerHTML = '<option value="">-- Select Device --</option>';
        
        devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.device_id;
            option.textContent = `${device.device_id} (${device.crop_type})`;
            option.dataset.cropType = device.crop_type;
            select.appendChild(option);
        });
        
        // Auto-select first device if available
        if (devices.length > 0) {
            select.value = devices[0].device_id;
            currentDevice = devices[0].device_id;
            loadDeviceData();
        }
    } catch (error) {
        console.error('Error loading devices:', error);
        showNotification('Failed to load devices', 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    document.getElementById('deviceSelect').addEventListener('change', function(e) {
        currentDevice = e.target.value;
        if (currentDevice) {
            loadDeviceData();
        }
    });
    
    document.getElementById('refreshBtn').addEventListener('click', () => loadDeviceData());
    document.getElementById('predictBtn').addEventListener('click', runPestPrediction);
    document.getElementById('activateBtn').addEventListener('click', () => controlIrrigation('activate'));
    document.getElementById('deactivateBtn').addEventListener('click', () => controlIrrigation('deactivate'));
}

// Load all data for current device
async function loadDeviceData() {
    if (!currentDevice) return;
    
    await Promise.all([
        loadLatestSensorData(),
        loadSensorHistory(),
        loadNDVIData()
    ]);
}

// Load latest sensor reading
async function loadLatestSensorData() {
    try {
        const response = await fetch(`${API_BASE}/sensors/${currentDevice}/latest`);
        const data = await response.json();
        
        document.getElementById('temperature').textContent = data.temperature.toFixed(1);
        document.getElementById('humidity').textContent = data.humidity.toFixed(1);
        
        const lastUpdate = new Date(data.timestamp * 1000);
        document.getElementById('lastUpdate').textContent = lastUpdate.toLocaleString();
        
    } catch (error) {
        console.error('Error loading sensor data:', error);
    }
}

// Load sensor history and update chart
async function loadSensorHistory() {
    try {
        const response = await fetch(`${API_BASE}/sensors/${currentDevice}/history?hours=24`);
        const history = await response.json();
        
        updateSensorChart(history);
        
    } catch (error) {
        console.error('Error loading sensor history:', error);
    }
}

// Update sensor chart
function updateSensorChart(data) {
    const ctx = document.getElementById('sensorChart').getContext('2d');
    
    const labels = data.map(d => new Date(d.timestamp * 1000).toLocaleTimeString());
    const temperatures = data.map(d => d.temperature);
    const humidities = data.map(d => d.humidity);
    
    if (sensorChart) {
        sensorChart.destroy();
    }
    
    sensorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Temperature (°C)',
                    data: temperatures,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Humidity (%)',
                    data: humidities,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f1f5f9'
                    }
                }
            },
            scales: {
                x: {
                    ticks: {
                        color: '#94a3b8',
                        maxTicksLimit: 8
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Temperature (°C)',
                        color: '#3b82f6'
                    },
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Humidity (%)',
                        color: '#10b981'
                    },
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

// Run pest prediction
async function runPestPrediction() {
    if (!currentDevice) {
        showNotification('Please select a device first', 'warning');
        return;
    }
    
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Analyzing...';
    
    try {
        const response = await fetch(`${API_BASE}/predict/pest?device_id=${currentDevice}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const prediction = await response.json();
        displayPestPrediction(prediction);
        generateRecommendations(prediction);
        showNotification('Pest prediction completed', 'success');
        
    } catch (error) {
        console.error('Error running prediction:', error);
        showNotification('Failed to run prediction. Ensure model is trained and sufficient data exists.', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🔍 Run Prediction';
    }
}

// Display pest prediction results
function displayPestPrediction(prediction) {
    const pred = prediction.prediction;
    const riskLevel = pred.risk_level;
    
    // Update risk indicator
    const indicator = document.getElementById('riskIndicator');
    indicator.className = `risk-indicator risk-${riskLevel.toLowerCase()}`;
    
    document.getElementById('riskLevel').textContent = riskLevel;
    document.getElementById('riskPercentage').textContent = `${pred.growth_percentage.toFixed(0)}%`;
    
    // Update alert message
    document.getElementById('alertMessage').textContent = prediction.alert.message;
    
    // Show timeline
    const timeline = document.getElementById('timeline');
    timeline.style.display = 'block';
    document.getElementById('timelineValue').textContent = `${pred.days_to_infestation} days`;
}

// Generate recommendations
function generateRecommendations(prediction) {
    const pred = prediction.prediction;
    const recommendations = document.getElementById('recommendations');
    
    let html = '<ul style="list-style: none; padding-left: 0;">';
    
    if (pred.risk_level === 'High' || pred.growth_percentage > 70) {
        html += '<li>🚨 <strong>Immediate Action Required:</strong></li>';
        html += '<li style="margin-left: 1.5rem;">• Apply targeted pesticide treatment within 24-48 hours</li>';
        html += '<li style="margin-left: 1.5rem;">• Increase field monitoring frequency</li>';
        html += '<li style="margin-left: 1.5rem;">• Consider professional consultation</li>';
    } else if (pred.risk_level === 'Medium' || pred.growth_percentage > 40) {
        html += '<li>⚠️ <strong>Preventive Measures:</strong></li>';
        html += '<li style="margin-left: 1.5rem;">• Prepare pesticide treatment equipment</li>';
        html += '<li style="margin-left: 1.5rem;">• Monitor environmental conditions daily</li>';
        html += '<li style="margin-left: 1.5rem;">• Check for early pest signs visually</li>';
    } else {
        html += '<li>✅ <strong>Routine Monitoring:</strong></li>';
        html += '<li style="margin-left: 1.5rem;">• Continue regular field inspections</li>';
        html += '<li style="margin-left: 1.5rem;">• Maintain optimal irrigation schedule</li>';
        html += '<li style="margin-left: 1.5rem;">• Review NDVI trends weekly</li>';
    }
    
    html += '</ul>';
    recommendations.innerHTML = html;
}

// Load NDVI data
async function loadNDVIData() {
    try {
        const plotId = currentDevice || 'plot_001';
        
        // Load current NDVI
        const currentResponse = await fetch(`${API_BASE}/ndvi/${plotId}/current`);
        const currentData = await currentResponse.json();
        
        document.getElementById('currentNDVI').textContent = currentData.current_ndvi.toFixed(3);
        
        // Load comparison
        const compareResponse = await fetch(`${API_BASE}/ndvi/${plotId}/compare?days=30`);
        const comparison = await compareResponse.json();
        
        // Update status
        document.getElementById('ndviStatus').innerHTML = `
            <p><strong>Status:</strong> ${comparison.vegetation_health_status}</p>
            <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem;">
                vs. Historical: ${comparison.comparison.percent_difference > 0 ? '+' : ''}${comparison.comparison.percent_difference.toFixed(1)}%
            </p>
        `;
        
        // Update NDVI chart (simplified - showing trend)
        updateNDVIChart(comparison);
        
    } catch (error) {
        console.error('Error loading NDVI data:', error);
        document.getElementById('ndviStatus').textContent = 'NDVI data unavailable';
    }
}

// Update NDVI chart
function updateNDVIChart(comparison) {
    const ctx = document.getElementById('ndviChart').getContext('2d');
    
    // Simulate simple visualization
    const labels = ['Historical Average', 'Current'];
    const data = [
        comparison.historical_baseline.mean_ndvi,
        comparison.current_period.mean_ndvi
    ];
    
    if (ndviChart) {
        ndviChart.destroy();
    }
    
    ndviChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'NDVI',
                data: data,
                backgroundColor: [
                    'rgba(132, 204, 22, 0.6)',
                    comparison.comparison.is_below_normal ? 'rgba(239, 68, 68, 0.6)' : 'rgba(34, 197, 94, 0.6)'
                ],
                borderColor: [
                    '#84cc16',
                    comparison.comparison.is_below_normal ? '#ef4444' : '#22c55e'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },
                x: {
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// Control irrigation
async function controlIrrigation(action) {
    if (!currentDevice) {
        showNotification('Please select a device first', 'warning');
        return;
    }
    
    const select = document.getElementById('deviceSelect');
    const cropType = select.selectedOptions[0].dataset.cropType || 'unknown';
    const duration = document.getElementById('durationInput').value;
    
    try {
        const endpoint = action === 'activate' 
            ? `${API_BASE}/irrigation/${currentDevice}/activate`
            : `${API_BASE}/irrigation/${currentDevice}/deactivate`;
        
        const body = {
            device_id: currentDevice,
            crop_type: cropType,
            command: action
        };
        
        if (action === 'activate') {
            body.duration_minutes = parseInt(duration);
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body)
        });
        
        const result = await response.json();
        
        // Update UI
        const valveState = document.getElementById('valveState');
        if (action === 'activate') {
            valveState.textContent = 'OPEN';
            valveState.style.color = '#22c55e';
            showNotification(`Valve activated for ${duration} minutes`, 'success');
        } else {
            valveState.textContent = 'CLOSED';
            valveState.style.color = '#94a3b8';
            showNotification('Valve deactivated', 'success');
        }
        
    } catch (error) {
        console.error('Error controlling irrigation:', error);
        showNotification('Failed to control irrigation', 'error');
    }
}

// WebSocket connection for real-time updates
function connectWebSocket() {
    try {
        ws = new WebSocket('ws://localhost:8000/ws/telemetry');
        
        ws.onopen = function() {
            console.log('WebSocket connected');
            document.getElementById('connectionStatus').className = 'status-online';
            document.getElementById('connectionStatus').textContent = '⚫ Online';
        };
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log('WebSocket message:', data);
            
            if (data.type === 'sensor_reading') {
                // Update display if it's for current device
                if (data.device_id === currentDevice) {
                    loadLatestSensorData();
                }
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
        ws.onclose = function() {
            console.log('WebSocket disconnected');
            document.getElementById('connectionStatus').className = 'status-offline';
            document.getElementById('connectionStatus').textContent = '⚫ Offline';
            
            // Attempt reconnect after 5 seconds
            setTimeout(connectWebSocket, 5000);
        };
    } catch (error) {
        console.error('Failed to create WebSocket:', error);
    }
}

// Show notification (simple toast)
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // In production, implement a proper toast notification system
}

// Auto-refresh every 30 seconds
setInterval(() => {
    if (currentDevice) {
        loadLatestSensorData();
    }
}, 30000);
