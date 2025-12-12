// Main JavaScript for Pneumonia Detection Demo

let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const resetBtn = document.getElementById('resetBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkModelsHealth();
});

function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });

    // Upload button
    uploadBtn.addEventListener('click', handleUpload);

    // Reset button
    resetBtn.addEventListener('click', resetDemo);
}

function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    if (!file.type.match('image.*')) {
        showError('Please select an image file');
        return;
    }

    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadArea.innerHTML = `
            <img src="${e.target.result}" class="preview" alt="Preview">
        `;
        uploadBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

async function handleUpload() {
    if (!selectedFile) return;

    // Hide upload section, show loading
    document.querySelector('.upload-section').style.display = 'none';
    results.style.display = 'none';
    loading.style.display = 'block';

    // Create form data
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to server');
    } finally {
        loading.style.display = 'none';
    }
}

function displayResults(data) {
    // Show image
    document.getElementById('previewImage').src = `data:image/png;base64,${data.image}`;

    // Display centralized model results
    displayModelResult('centralized', data.centralized);

    // Display federated model results
    displayModelResult('federated', data.federated);

    // Check agreement
    displayAgreement(data.centralized, data.federated);

    // Show results section
    results.style.display = 'block';
    
    // Scroll to results
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayModelResult(modelType, result) {
    const prefix = modelType;

    // Set class and icon
    const classElement = document.getElementById(`${prefix}Class`);
    const iconElement = document.getElementById(`${prefix}Icon`);
    const confidenceElement = document.getElementById(`${prefix}Confidence`);

    classElement.textContent = result.class;
    confidenceElement.textContent = `Confidence: ${result.confidence.toFixed(2)}%`;

    // Update icon
    iconElement.className = `result-icon ${result.class.toLowerCase()}`;
    if (result.class === 'Normal') {
        iconElement.innerHTML = '<i class="fas fa-check"></i>';
    } else {
        iconElement.innerHTML = '<i class="fas fa-exclamation"></i>';
    }

    // Update probability bars
    document.getElementById(`${prefix}PneumoniaProb`).textContent = 
        `${result.pneumonia_probability.toFixed(1)}%`;
    document.getElementById(`${prefix}PneumoniaBar`).style.width = 
        `${result.pneumonia_probability}%`;

    document.getElementById(`${prefix}NormalProb`).textContent = 
        `${result.normal_probability.toFixed(1)}%`;
    document.getElementById(`${prefix}NormalBar`).style.width = 
        `${result.normal_probability}%`;
}

function displayAgreement(centralized, federated) {
    const agreementCard = document.getElementById('agreement');
    const agreementText = document.getElementById('agreementText');

    if (centralized.class === federated.class) {
        agreementCard.className = 'agreement-card';
        agreementCard.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <p>Both models agree: <strong>${centralized.class}</strong> detected</p>
        `;
    } else {
        agreementCard.className = 'agreement-card disagree';
        agreementCard.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <p>Models disagree - Centralized: <strong>${centralized.class}</strong>, Federated: <strong>${federated.class}</strong></p>
        `;
    }
}

function resetDemo() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.innerHTML = `
        <div class="upload-prompt">
            <i class="fas fa-image"></i>
            <p>Click to upload or drag and drop</p>
            <span>PNG, JPG, JPEG (Max 16MB)</span>
        </div>
    `;
    uploadBtn.disabled = true;
    results.style.display = 'none';
    document.querySelector('.upload-section').style.display = 'block';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function showError(message) {
    alert(`Error: ${message}`);
    resetDemo();
}

async function checkModelsHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.models_loaded) {
            console.log('Models are still loading...');
            setTimeout(checkModelsHealth, 2000);
        } else {
            console.log('✅ Models loaded and ready');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Utility function to format percentage
function formatPercent(value) {
    return `${value.toFixed(1)}%`;
}
