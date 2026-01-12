/**
 * OCR Model Comparison Frontend Application
 */

const API_BASE = '/api';

// State
let selectedModels = new Set(['mistral-large-3', 'deepseek-ocr-gpu', 'gpt-5.2', 'gemini-3-flash', 'azure-doc-intelligence']);
let uploadedImage = null;
let imageType = 'image/png';

// Model display info
const modelInfo = {
    'mistral-large-3': {
        name: 'Mistral Large 3',
        icon: '🌀',
        color: '#ff6b35'
    },
    'deepseek-ocr-gpu': {
        name: 'DeepSeek OCR (Azure GPU)',
        icon: '🚀',
        color: '#059669'
    },
    'gpt-5.2': {
        name: 'GPT-5.2',
        icon: '🤖',
        color: '#8b5cf6'
    },
    'gemini-3-flash': {
        name: 'Gemini 3 Flash',
        icon: '✨',
        color: '#4285f4'
    },
    'azure-doc-intelligence': {
        name: 'Azure Doc Intelligence',
        icon: '📄',
        color: '#0078d4'
    }
};

// DOM Elements
const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const imageInfo = document.getElementById('imageInfo');
const modelCards = document.getElementById('modelCards');
const compareBtn = document.getElementById('compareBtn');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingStatus = document.getElementById('loadingStatus');
const resultsSection = document.getElementById('resultsSection');
const summaryCards = document.getElementById('summaryCards');
const resultsGrid = document.getElementById('resultsGrid');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeModelCards();
    setupEventListeners();
});

function initializeModelCards() {
    const models = [
        {
            id: 'mistral-large-3',
            name: 'Mistral Large 3',
            pricing: { input_per_1k: 0.002, output_per_1k: 0.006 }
        },
        {
            id: 'deepseek-ocr-gpu',
            name: 'DeepSeek OCR (Azure GPU)',
            pricing: { input_per_1k: 0.002, output_per_1k: 0.002 }
        },
        {
            id: 'gpt-5.2',
            name: 'GPT-5.2',
            pricing: { input_per_1k: 0.01, output_per_1k: 0.03 }
        },
        {
            id: 'gemini-3-flash',
            name: 'Gemini 3 Flash',
            pricing: { input_per_1k: 0.00001, output_per_1k: 0.00004 }
        },
        {
            id: 'azure-doc-intelligence',
            name: 'Azure Doc Intelligence',
            pricing: { input_per_1k: 0.001, output_per_1k: 0.0 }
        }
    ];

    modelCards.innerHTML = models.map(model => `
        <div class="model-card ${selectedModels.has(model.id) ? 'selected' : ''}" 
             data-model-id="${model.id}">
            <div class="check-icon">
                <i class="fas fa-check"></i>
            </div>
            <h3>${modelInfo[model.id]?.icon || '🤖'} ${model.name}</h3>
            <div class="pricing">
                <span>Input: $${model.pricing.input_per_1k.toFixed(5)}/1K tokens</span>
                <span>Output: $${model.pricing.output_per_1k.toFixed(5)}/1K tokens</span>
            </div>
        </div>
    `).join('');

    // Add click listeners to model cards
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', () => toggleModel(card.dataset.modelId));
    });
}

function setupEventListeners() {
    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImageUpload(file);
        }
    });

    // File input
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });

    // Click on drop zone (but not on the button which is handled separately)
    dropZone.addEventListener('click', (e) => {
        // Prevent double trigger when clicking the button
        if (e.target.closest('.select-image-btn')) {
            return;
        }
        imageInput.click();
    });

    // Click on select image button
    const selectBtn = document.querySelector('.select-image-btn');
    if (selectBtn) {
        selectBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent drop zone click handler
            imageInput.click();
        });
    }

    // Compare button
    compareBtn.addEventListener('click', runComparison);
}

function toggleModel(modelId) {
    const card = document.querySelector(`[data-model-id="${modelId}"]`);
    
    if (selectedModels.has(modelId)) {
        selectedModels.delete(modelId);
        card.classList.remove('selected');
    } else {
        selectedModels.add(modelId);
        card.classList.add('selected');
    }
    
    updateCompareButton();
}

function handleImageUpload(file) {
    imageType = file.type;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage = e.target.result;
        imagePreview.src = uploadedImage;
        previewContainer.style.display = 'block';
        
        // Display image info
        const sizeKB = (file.size / 1024).toFixed(2);
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        imageInfo.innerHTML = `
            <strong>File:</strong> ${file.name}<br>
            <strong>Type:</strong> ${file.type}<br>
            <strong>Size:</strong> ${sizeMB > 1 ? sizeMB + ' MB' : sizeKB + ' KB'}
        `;
        
        updateCompareButton();
    };
    reader.readAsDataURL(file);
}

function updateCompareButton() {
    const canCompare = uploadedImage && selectedModels.size > 0;
    compareBtn.disabled = !canCompare;
}

async function runComparison() {
    if (!uploadedImage || selectedModels.size === 0) return;
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    loadingStatus.textContent = 'Sending image to OCR models...';
    resultsSection.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/ocr`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: uploadedImage,
                image_type: imageType,
                models: Array.from(selectedModels)
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data.results);
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error processing OCR: ${error.message}`);
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

function displayResults(results) {
    resultsSection.style.display = 'block';
    
    // Find winners in each category
    let bestAccuracyModel = null;
    let highestQuality = -1;
    let fastestModel = null;
    let fastestTime = Infinity;
    let cheapestModel = null;
    let cheapestCost = Infinity;
    
    Object.entries(results).forEach(([modelId, result]) => {
        if (result.status === 'success') {
            // Best accuracy (highest quality score)
            if (result.metrics.quality_score > highestQuality) {
                highestQuality = result.metrics.quality_score;
                bestAccuracyModel = modelId;
            }
            // Fastest (lowest response time)
            if (result.metrics.response_time > 0 && result.metrics.response_time < fastestTime) {
                fastestTime = result.metrics.response_time;
                fastestModel = modelId;
            }
            // Most cost effective (lowest cost, but exclude free/zero cost for fair comparison)
            const cost = result.metrics.cost_usd;
            if (cost < cheapestCost) {
                cheapestCost = cost;
                cheapestModel = modelId;
            }
        }
    });
    
    // Display summary cards - highlighting 3 categories
    summaryCards.innerHTML = `
        <div class="summary-card accuracy">
            <div class="icon">🎯</div>
            <div class="label">Best Accuracy</div>
            <div class="value">${bestAccuracyModel ? modelInfo[bestAccuracyModel]?.name || bestAccuracyModel : 'N/A'}</div>
            <div class="detail">${highestQuality > 0 ? `Score: ${highestQuality}/100` : ''}</div>
        </div>
        <div class="summary-card speed">
            <div class="icon">⚡</div>
            <div class="label">Fastest</div>
            <div class="value">${fastestModel ? modelInfo[fastestModel]?.name || fastestModel : 'N/A'}</div>
            <div class="detail">${fastestTime < Infinity ? `${fastestTime.toFixed(2)}s` : ''}</div>
        </div>
        <div class="summary-card cost">
            <div class="icon">💰</div>
            <div class="label">Most Cost Effective</div>
            <div class="value">${cheapestModel ? modelInfo[cheapestModel]?.name || cheapestModel : 'N/A'}</div>
            <div class="detail">${cheapestCost < Infinity ? (cheapestCost === 0 ? 'FREE' : `$${cheapestCost.toFixed(6)}`) : ''}</div>
        </div>
        <div class="summary-card">
            <div class="icon">📊</div>
            <div class="label">Models Compared</div>
            <div class="value">${Object.keys(results).length}</div>
        </div>
    `;
    
    // Display detailed results
    resultsGrid.innerHTML = Object.entries(results).map(([modelId, result]) => {
        const info = modelInfo[modelId] || { name: modelId, icon: '🤖', color: '#6366f1' };
        const isBestAccuracy = modelId === bestAccuracyModel;
        const isFastest = modelId === fastestModel;
        const isCheapest = modelId === cheapestModel;
        
        if (result.status === 'error') {
            const errorDebugLogs = result.debug_logs || [];
            const errorDebugLogsHtml = errorDebugLogs.length > 0 
                ? `<div class="debug-logs-container" id="debug-${modelId}" style="display: none;">
                    <div class="debug-logs-content">
                        ${errorDebugLogs.map(log => `<div class="debug-log-line">${escapeHtml(log)}</div>`).join('')}
                    </div>
                   </div>`
                : '';
            
            return `
                <div class="result-card error-card">
                    <div class="result-header">
                        <h3>${info.icon} ${info.name}</h3>
                    </div>
                    <div class="result-error">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Error: ${result.error}</p>
                    </div>
                    ${errorDebugLogs.length > 0 ? `
                    <div class="debug-section">
                        <button class="debug-btn" onclick="toggleDebugLogs('${modelId}')">
                            <i class="fas fa-terminal"></i> More Details
                        </button>
                        ${errorDebugLogsHtml}
                    </div>
                    ` : ''}
                </div>
            `;
        }
        
        const m = result.metrics;
        
        // Build badges for each award
        const badges = [];
        if (isBestAccuracy) badges.push('<span class="badge accuracy-badge">🎯 Best Accuracy</span>');
        if (isFastest) badges.push('<span class="badge speed-badge">⚡ Fastest</span>');
        if (isCheapest) badges.push('<span class="badge cost-badge">💰 Most Cost Effective</span>');
        const badgesHtml = badges.length > 0 ? `<div class="badges">${badges.join('')}</div>` : '';
        
        // Card gets special styling if it has any award
        const hasAward = isBestAccuracy || isFastest || isCheapest;
        
        // Calculate circular progress for quality score
        const qualityScore = m.quality_score;
        const circumference = 2 * Math.PI * 36; // radius = 36
        const strokeDashoffset = circumference - (qualityScore / 100) * circumference;
        const scoreColor = qualityScore >= 80 ? '#22c55e' : qualityScore >= 60 ? '#eab308' : qualityScore >= 40 ? '#f97316' : '#ef4444';
        
        // Build debug logs HTML
        const debugLogs = result.debug_logs || [];
        const debugLogsHtml = debugLogs.length > 0 
            ? `<div class="debug-logs-container" id="debug-${modelId}" style="display: none;">
                <div class="debug-logs-content">
                    ${debugLogs.map(log => `<div class="debug-log-line">${escapeHtml(log)}</div>`).join('')}
                </div>
               </div>`
            : '';
        
        return `
            <div class="result-card ${hasAward ? 'has-award' : ''}">
                <div class="result-header">
                    <div class="header-content">
                        <h3>${info.icon} ${info.name}</h3>
                        <div class="quality-circle-container">
                            <div class="quality-circle">
                                <svg width="80" height="80" viewBox="0 0 80 80">
                                    <circle cx="40" cy="40" r="36" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="6"/>
                                    <circle cx="40" cy="40" r="36" fill="none" stroke="${scoreColor}" stroke-width="6" 
                                        stroke-linecap="round"
                                        stroke-dasharray="${circumference}" 
                                        stroke-dashoffset="${strokeDashoffset}"
                                        transform="rotate(-90 40 40)"/>
                                </svg>
                                <div class="quality-score-value">${qualityScore}</div>
                            </div>
                            <div class="quality-label">Quality Score</div>
                        </div>
                    </div>
                    ${badgesHtml}
                </div>
                <div class="result-metrics">
                    <div class="metric ${isBestAccuracy ? 'highlight-accuracy' : ''}">
                        <div class="metric-label">Quality Score</div>
                        <div class="metric-value">${m.quality_score}/100</div>
                    </div>
                    <div class="metric ${isFastest ? 'highlight-speed' : ''}">
                        <div class="metric-label">Response Time</div>
                        <div class="metric-value">${m.response_time}s</div>
                    </div>
                    <div class="metric ${isCheapest ? 'highlight-cost' : ''}">
                        <div class="metric-label">Estimated Cost</div>
                        <div class="metric-value">${m.cost_usd === 0 ? 'FREE' : '$' + m.cost_usd.toFixed(6)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Input Tokens</div>
                        <div class="metric-value">${m.input_tokens.toLocaleString()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Output Tokens</div>
                        <div class="metric-value">${m.output_tokens.toLocaleString()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Total Tokens</div>
                        <div class="metric-value">${m.total_tokens.toLocaleString()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Characters</div>
                        <div class="metric-value">${m.char_count.toLocaleString()}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Words</div>
                        <div class="metric-value">${m.word_count.toLocaleString()}</div>
                    </div>
                </div>
                <div class="result-text">
                    <h4><i class="fas fa-file-alt"></i> Extracted Text</h4>
                    <div class="text-output">${escapeHtml(result.text) || 'No text extracted'}</div>
                </div>
                ${debugLogs.length > 0 ? `
                <div class="debug-section">
                    <button class="debug-btn" onclick="toggleDebugLogs('${modelId}')">
                        <i class="fas fa-terminal"></i> More Details
                    </button>
                    ${debugLogsHtml}
                </div>
                ` : ''}
            </div>
        `;
    }).join('');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Toggle debug logs visibility
function toggleDebugLogs(modelId) {
    const debugContainer = document.getElementById(`debug-${modelId}`);
    const btn = debugContainer.previousElementSibling;
    
    if (debugContainer.style.display === 'none') {
        debugContainer.style.display = 'block';
        btn.innerHTML = '<i class="fas fa-terminal"></i> Hide Details';
        btn.classList.add('active');
    } else {
        debugContainer.style.display = 'none';
        btn.innerHTML = '<i class="fas fa-terminal"></i> More Details';
        btn.classList.remove('active');
    }
}

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}
