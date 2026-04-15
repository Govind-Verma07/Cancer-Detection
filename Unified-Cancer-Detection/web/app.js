// Tab Switching Logic
function switchTab(tabId) {
    // Update tab UI
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tab-${tabId}`).classList.add('active');

    // Update view UI
    document.querySelectorAll('.view').forEach(view => view.classList.remove('active'));
    
    // Slight delay for animation flow
    setTimeout(() => {
        document.querySelectorAll('.view').forEach(view => view.classList.add('hidden'));
        const activeView = document.getElementById(`view-${tabId}`);
        activeView.classList.remove('hidden');
        activeView.classList.add('active');
        
        // If switching to analytics, trigger fetch
        if (tabId === 'analytics') {
            loadAnalytics();
        }
    }, 150);
}

// Drag and Drop Logic
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadDetails = document.getElementById('upload-details');
const fileNameDisplay = document.getElementById('file-name');
let currentFile = null;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
});

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        currentFile = files[0];
        
        // Ensure it's an image
        if (!currentFile.type.startsWith('image/')) {
            alert('Please upload an image file (JPEG/PNG).');
            return;
        }

        fileNameDisplay.textContent = `📎 ${currentFile.name} (${(currentFile.size / 1024).toFixed(2)} KB)`;
        uploadDetails.classList.remove('hidden');
        document.getElementById('results-section').classList.add('hidden');
    }
}

// Inference Orchestration
async function runInference() {
    if (!currentFile) {
        alert("Please select a file first.");
        return;
    }

    const groundTruth = document.getElementById('ground-truth').value;
    
    // UI State
    dropZone.parentElement.classList.add('hidden');
    document.getElementById('loader').classList.remove('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('conclusion-section').classList.add('hidden');

    // Build Payload
    const formData = new FormData();
    formData.append("file", currentFile);
    if (groundTruth !== "") {
        formData.append("ground_truth", groundTruth);
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("API Request Failed");

        const data = await response.json();
        populateResults(data);

    } catch (error) {
        console.error("Inference Error:", error);
        alert("Error running inference. Ensure backend is running.");
        // Revert UI
        document.getElementById('loader').classList.add('hidden');
        dropZone.parentElement.classList.remove('hidden');
    }
}

// Result Parsing
function populateResults(data) {
    document.getElementById('loader').classList.add('hidden');
    dropZone.parentElement.classList.remove('hidden');
    
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');
    
    const conclusionSection = document.getElementById('conclusion-section');
    conclusionSection.classList.remove('hidden');

    // Populate ResNet
    document.getElementById('resnet-img').src = data.resnet.image_url || '/media/placeholder_mask.png';
    document.getElementById('resnet-burden').textContent = `${(data.resnet.tumor_burden * 100).toFixed(2)}%`;
    document.getElementById('resnet-zones').textContent = data.resnet.regions_detected;

    // Populate VGG
    document.getElementById('vgg-img').src = data.vgg.image_url || '/media/placeholder_mask.png';
    document.getElementById('vgg-burden').textContent = `${(data.vgg.tumor_burden * 100).toFixed(2)}%`;
    document.getElementById('vgg-zones').textContent = data.vgg.regions_detected;

    // Populate Consensus
    const ensembleScore = (data.ensemble.score * 100).toFixed(2);
    document.getElementById('ensemble-score').textContent = `${ensembleScore}%`;
    
    const statusEl = document.getElementById('final-status');
    statusEl.textContent = data.ensemble.status;
    
    // Status colors
    statusEl.className = '';
    if (data.ensemble.status === 'Malignant') {
        statusEl.classList.add('status-malignant');
    } else if (data.ensemble.status === 'Benign') {
        statusEl.classList.add('status-benign');
    } else {
        statusEl.classList.add('status-review');
    }
    
    // Populate Conclusion Report
    document.getElementById('conclusion-report').value = data.ensemble.conclusion_report || 'No conclusion available';
}

// Chart Instance
let accuracyChartInstance = null;

// Analytics Loading
async function loadAnalytics() {
    try {
        // Fetch History
        const histRes = await fetch('/api/history');
        const historyData = await histRes.json();
        populateHistoryTable(historyData);

        // Fetch Analytics Series
        const accRes = await fetch('/api/analytics');
        const accData = await accRes.json();
        renderChart(accData);

    } catch (error) {
        console.error("Failed to fetch analytics", error);
    }
}

function populateHistoryTable(data) {
    const tbody = document.getElementById('history-body');
    tbody.innerHTML = '';
    
    if (data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding: 2rem;">No inference history available yet.</td></tr>';
        return;
    }

    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.timestamp || 'N/A'}</td>
            <td>${row.filename || 'Unknown'}</td>
            <td>${(row.resnet_score * 100).toFixed(1)}%</td>
            <td>${(row.vgg_score * 100).toFixed(1)}%</td>
            <td><strong>${(row.ensemble_score * 100).toFixed(1)}%</strong></td>
            <td><span class="${row.status === 'Malignant' ? 'status-malignant' : 'status-benign'}">${row.status}</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function renderChart(data) {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    
    if (accuracyChartInstance) {
        accuracyChartInstance.destroy();
    }

    const labels = data.map((_, idx) => `Test ${idx + 1}`);
    const resnetAcc = data.map(d => d.resnet_accuracy * 100);
    const vggAcc = data.map(d => d.vgg_accuracy * 100);

    accuracyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'ResNet50 Accuracy (%)',
                    data: resnetAcc,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'VGG16 Accuracy (%)',
                    data: vggAcc,
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Accuracy Test Function
async function runAccuracyTest() {
    const statusEl = document.getElementById('accuracy-status');
    statusEl.classList.remove('hidden');
    statusEl.textContent = 'Running accuracy test on 100 images...';
    statusEl.className = 'status-message';

    try {
        const response = await fetch('/api/run_accuracy_test', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ num_images: 100 })
        });

        if (!response.ok) throw new Error("Accuracy test failed");

        const result = await response.json();
        statusEl.textContent = 'Accuracy test completed successfully!';
        statusEl.classList.add('success');
        
        // Reload analytics to show updated data
        loadAnalytics();
        
    } catch (error) {
        console.error("Accuracy test error:", error);
        statusEl.textContent = 'Error running accuracy test. Check console for details.';
        statusEl.classList.add('error');
    }
}
