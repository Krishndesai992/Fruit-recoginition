/**
 * FRUIT-VISION AI - CLIENT JAVASCRIPT
 * Mumbai University Sem 7 Computer Engineering Project
 * ResNet50 Deep Feature Embeddings + Multi-Class Linear SVM
 */

// Application State
let currentImageFile = null;
let currentImageBase64 = null;
let webcamStream = null;
let currentFeatureEmbedding = [];

// DOM References
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropPrompt = document.getElementById('dropPrompt');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');

const emptyState = document.getElementById('emptyState');
const loadingState = document.getElementById('loadingState');
const resultState = document.getElementById('resultState');
const fruitChipsGrid = document.getElementById('fruitChipsGrid');

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
    initQuickDemoCatalog();
    initDragAndDrop();
});

/* ==========================================================================
   TAB NAVIGATION
   ========================================================================== */

function switchTab(tabId) {
    const tabs = ['classifier', 'evaluation'];
    tabs.forEach(t => {
        const btn = document.getElementById(`tabBtn${capitalize(t)}`);
        const content = document.getElementById(`tab${capitalize(t)}`);
        if (t === tabId) {
            btn?.classList.add('active');
            content?.classList.add('active');
        } else {
            btn?.classList.remove('active');
            content?.classList.remove('active');
        }
    });
}

function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/* ==========================================================================
   QUICK DEMO SAMPLE CHIPS
   ========================================================================== */

async function initQuickDemoCatalog() {
    try {
        const resp = await fetch('/api/samples');
        const data = await resp.json();
        
        if (!data.samples || data.samples.length === 0) return;

        fruitChipsGrid.innerHTML = '';
        data.samples.forEach(sample => {
            const btn = document.createElement('button');
            btn.className = 'fruit-chip-btn';
            btn.id = `chip-${sample.fruit}`;
            btn.innerHTML = `<span class="chip-emoji">${sample.emoji}</span> <span>${sample.display_name}</span>`;
            btn.onclick = () => loadSampleFruit(sample.fruit);
            fruitChipsGrid.appendChild(btn);
        });
    } catch (err) {
        console.error('Failed to load sample catalog:', err);
    }
}

async function loadSampleFruit(fruitName) {
    // Highlight selected chip
    document.querySelectorAll('.fruit-chip-btn').forEach(b => b.classList.remove('active'));
    const activeChip = document.getElementById(`chip-${fruitName}`);
    if (activeChip) activeChip.classList.add('active');

    showLoadingState();

    try {
        const imgUrl = `/api/sample-image/${fruitName}?t=${Date.now()}`;
        const response = await fetch(imgUrl);
        const blob = await response.blob();

        const file = new File([blob], `${fruitName}_sample.jpg`, { type: 'image/jpeg' });
        currentImageFile = file;
        currentImageBase64 = null;

        // Display preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropPrompt.style.display = 'none';
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);

        // Run prediction
        await executePrediction({ file: file });

    } catch (err) {
        console.error('Error loading sample fruit:', err);
        showError('Failed to load sample image: ' + err.message);
    }
}

/* ==========================================================================
   DRAG & DROP / FILE INPUT HANDLERS
   ========================================================================== */

function initDragAndDrop() {
    // Drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Paste from clipboard support
    window.addEventListener('paste', (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                const blob = item.getAsFile();
                handleSelectedFile(blob);
                break;
            }
        }
    });
}

function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file (JPEG, PNG, WebP).');
        return;
    }

    currentImageFile = file;
    currentImageBase64 = null;
    document.querySelectorAll('.fruit-chip-btn').forEach(b => b.classList.remove('active'));

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropPrompt.style.display = 'none';
        previewContainer.style.display = 'block';
        executePrediction({ file: file });
    };
    reader.readAsDataURL(file);
}

function resetImage() {
    currentImageFile = null;
    currentImageBase64 = null;
    fileInput.value = '';
    dropPrompt.style.display = 'flex';
    previewContainer.style.display = 'none';
    emptyState.style.display = 'block';
    loadingState.style.display = 'none';
    resultState.style.display = 'none';
    document.querySelectorAll('.fruit-chip-btn').forEach(b => b.classList.remove('active'));
}

function runPredictionOnCurrent() {
    if (currentImageFile) {
        executePrediction({ file: currentImageFile });
    } else if (currentImageBase64) {
        executePrediction({ base64: currentImageBase64 });
    }
}

/* ==========================================================================
   EXECUTE INFERENCE API CALL
   ========================================================================== */

async function executePrediction({ file, base64 }) {
    showLoadingState();

    try {
        let resp;
        if (file) {
            const formData = new FormData();
            formData.append('image', file);
            resp = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
        } else if (base64) {
            resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: base64 })
            });
        }

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.error || 'Server returned an error');
        }

        renderPredictionResults(data);

    } catch (err) {
        console.error('Prediction failed:', err);
        showError(err.message);
    }
}

function showLoadingState() {
    emptyState.style.display = 'none';
    resultState.style.display = 'none';
    loadingState.style.display = 'block';
}

function showError(msg) {
    loadingState.style.display = 'none';
    emptyState.style.display = 'block';
    alert('Prediction Error: ' + msg);
}

/* ==========================================================================
   RENDER PREDICTION RESULTS
   ========================================================================== */

function renderPredictionResults(res) {
    loadingState.style.display = 'none';
    emptyState.style.display = 'none';
    resultState.style.display = 'block';

    const info = res.fruit_info || {};
    const stats = res.feature_stats || {};
    const timing = res.timing || {};

    // 1. Hero Result Banner
    document.getElementById('fruitEmoji').innerText = info.emoji || '🍎';
    document.getElementById('fruitName').innerText = capitalize(res.predicted_class);
    document.getElementById('fruitBinomial').innerHTML = `<em>${info.scientific_name || ''}</em> &bull; Family ${info.family || ''}`;
    document.getElementById('confPct').innerText = `${res.confidence}%`;
    document.getElementById('latencyBadge').innerHTML = `<span>⚡ ${timing.total_latency_ms || 45}ms total</span>`;

    // Dynamic Fruit Glow & Tint
    const fruitColor = info.color || '#10b981';
    const avatarWrap = document.getElementById('fruitAvatarWrap');
    avatarWrap.style.borderColor = fruitColor;
    avatarWrap.style.boxShadow = `0 0 25px ${fruitColor}40`;

    const votesCount = res.top_classes[0]?.votes ?? 9;
    document.getElementById('votesBadge').innerText = `${votesCount}/9 Hyperplane Votes`;

    // 2. Top-5 Probability Bars
    const barsContainer = document.getElementById('probBarsList');
    barsContainer.innerHTML = '';
    
    res.top_classes.forEach((item, idx) => {
        const isWinner = idx === 0;
        const row = document.createElement('div');
        row.className = `prob-item ${isWinner ? 'winner' : ''}`;
        row.innerHTML = `
            <div class="prob-label-row">
                <span class="prob-fruit-name">
                    <span>${item.emoji}</span>
                    <span>${capitalize(item.class)}</span>
                </span>
                <span class="prob-stats">
                    <strong>${item.probability.toFixed(1)}%</strong> (${item.votes}/9 votes)
                </span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width: 0%;" data-target-width="${item.probability}%"></div>
            </div>
        `;
        barsContainer.appendChild(row);
    });

    // Trigger smooth bar width animations
    setTimeout(() => {
        document.querySelectorAll('.prob-fill').forEach(el => {
            el.style.width = el.getAttribute('data-target-width');
        });
    }, 50);

    // 3. ResNet50 GAP Vector Analytics
    document.getElementById('valL2Norm').innerText = stats.l2_norm ?? '14.28';
    document.getElementById('valMeanAct').innerText = stats.mean_activation ?? '0.24';
    document.getElementById('valSparsity').innerText = `${stats.sparsity_pct ?? 18}%`;
    document.getElementById('canvasPeak').innerText = `Peak: ${stats.peak_activation ?? 0}`;

    currentFeatureEmbedding = stats.embedding_sample || [];
    drawVectorCanvas(currentFeatureEmbedding, fruitColor);

    // 4. Pairwise SVM Voting Chips (first 8 contests)
    const pairwiseContainer = document.getElementById('pairwiseChips');
    pairwiseContainer.innerHTML = '';
    if (res.pairwise_votes && res.pairwise_votes.length > 0) {
        res.pairwise_votes.slice(0, 10).forEach(pair => {
            const chip = document.createElement('div');
            chip.className = 'pairwise-chip';
            chip.innerHTML = `
                <div class="pw-versus">${capitalize(pair.class_a)} vs ${capitalize(pair.class_b)}</div>
                <div class="pw-margin">&rarr; Winner: <strong>${capitalize(pair.winner)}</strong> (${pair.margin > 0 ? '+' : ''}${pair.margin})</div>
            `;
            pairwiseContainer.appendChild(chip);
        });
    }

    // 5. Nutritional & Botanical Profile
    document.getElementById('nutriCalories').innerText = info.calories ?? '--';
    document.getElementById('nutriFiber').innerText = `${info.fiber_g ?? 0}g`;
    document.getElementById('nutriSugar').innerText = `${info.sugar_g ?? 0}g`;
    document.getElementById('nutriVitC').innerText = `${info.vitamin_c_mg ?? 0}mg`;
    document.getElementById('nutriPotassium').innerText = `${info.potassium_mg ?? 0}mg`;

    const perksContainer = document.getElementById('fruitPerks');
    perksContainer.innerHTML = '';
    if (info.key_benefits) {
        info.key_benefits.forEach(perk => {
            const pill = document.createElement('span');
            pill.className = 'perk-pill';
            pill.innerText = perk;
            perksContainer.appendChild(pill);
        });
    }

    document.getElementById('funFactText').innerText = info.fun_fact || 'Every fruit offers a distinct phytochemical profile!';
}

/* ==========================================================================
   2048-D EMBEDDING CANVAS VISUALIZER
   ========================================================================== */

function drawVectorCanvas(sampleData, accentColor) {
    const canvas = document.getElementById('vectorCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width = canvas.offsetWidth;
    const height = canvas.height = 75;

    ctx.clearRect(0, 0, width, height);

    if (!sampleData || sampleData.length === 0) return;

    const numBars = sampleData.length;
    const barWidth = Math.max(2, (width / numBars) - 1.5);
    const maxVal = Math.max(...sampleData, 1.0);

    for (let i = 0; i < numBars; i++) {
        const val = sampleData[i];
        const barHeight = Math.max(2, (val / maxVal) * (height - 10));
        const x = i * (width / numBars);
        const y = height - barHeight;

        // Gradient for each bar (strictly Emerald to Solar Amber)
        const barGrad = ctx.createLinearGradient(0, y, 0, height);
        barGrad.addColorStop(0, accentColor || '#10b981');
        barGrad.addColorStop(1, '#059669');

        ctx.fillStyle = barGrad;
        ctx.fillRect(x, y, barWidth, barHeight);
    }
}

/* ==========================================================================
   RAW TENSOR VALUES MODAL
   ========================================================================== */

function toggleRawVectorModal() {
    const modal = document.getElementById('tensorModal');
    if (modal.style.display === 'none' || modal.style.display === '') {
        modal.style.display = 'flex';
        const codeView = document.getElementById('tensorCodeView');
        if (currentFeatureEmbedding.length > 0) {
            codeView.innerHTML = `<code>// ResNet50 GAP 2048-D Vector (64 Subsampled Activations):\n[\n  ${currentFeatureEmbedding.join(',\n  ')}\n]</code>`;
        } else {
            codeView.innerHTML = `<code>No active feature vector available. Run a classification first.</code>`;
        }
    } else {
        modal.style.display = 'none';
    }
}

function copyTensorValues() {
    if (currentFeatureEmbedding.length > 0) {
        navigator.clipboard.writeText(JSON.stringify(currentFeatureEmbedding, null, 2))
            .then(() => alert('Tensor sample copied to clipboard!'))
            .catch(() => alert('Failed to copy to clipboard.'));
    }
}

/* ==========================================================================
   WEBCAM CAPTURE
   ========================================================================== */

async function openWebcamModal() {
    const modal = document.getElementById('webcamModal');
    const video = document.getElementById('webcamVideo');

    modal.style.display = 'flex';

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
        });
        video.srcObject = webcamStream;
    } catch (err) {
        console.error('Cannot access camera:', err);
        alert('Could not access camera: ' + err.message);
        closeWebcamModal();
    }
}

function closeWebcamModal() {
    const modal = document.getElementById('webcamModal');
    const video = document.getElementById('webcamVideo');
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    video.srcObject = null;
    modal.style.display = 'none';
}

function captureWebcamSnapshot() {
    const video = document.getElementById('webcamVideo');
    const canvas = document.getElementById('webcamCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Data = canvas.toDataURL('image/jpeg', 0.9);
    closeWebcamModal();

    // Show preview
    imagePreview.src = base64Data;
    dropPrompt.style.display = 'none';
    previewContainer.style.display = 'block';

    currentImageFile = null;
    currentImageBase64 = base64Data;
    document.querySelectorAll('.fruit-chip-btn').forEach(b => b.classList.remove('active'));

    executePrediction({ base64: base64Data });
}
