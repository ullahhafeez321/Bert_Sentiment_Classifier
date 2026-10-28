/* =====================================================
   BERT Sentiment Classifier — App Logic
   Hugging Face Inference API Integration
   ===================================================== */

// ─── Config ───
const HF_MODEL = 'ullahhafeez321/bert-sentiment-classifier';
const HF_API_URL = `https://api-inference.huggingface.co/models/${HF_MODEL}`;
const MAX_CHARS = 2000;
const HISTORY_KEY = 'bert_sentiment_history';
const TOKEN_KEY = 'bert_hf_token';

// ─── DOM Refs ───
const textInput = document.getElementById('text-input');
const charCount = document.getElementById('char-count');
const btnAnalyze = document.getElementById('btn-analyze');
const btnClear = document.getElementById('btn-clear');
const resultsSection = document.getElementById('results-section');
const historySection = document.getElementById('history-section');
const historyGrid = document.getElementById('history-grid');
const apiPanel = document.getElementById('api-panel');
const apiTokenInput = document.getElementById('api-token-input');
const apiStatus = document.getElementById('api-status');
const apiDot = document.getElementById('api-dot');

// ─── State ───
let isAnalyzing = false;
let history = [];

// ─── Initialization ───
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    loadTokenStatus();
    textInput.addEventListener('input', updateCharCount);
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            analyzeSentiment();
        }
    });
});

// ─── Character Count ───
function updateCharCount() {
    const len = textInput.value.length;
    charCount.textContent = `${len.toLocaleString()} / 2,000`;
    if (len > MAX_CHARS * 0.9) {
        charCount.style.color = 'var(--negative-color)';
    } else {
        charCount.style.color = '';
    }
}

// ─── API Panel Toggle ───
function toggleApiPanel() {
    apiPanel.classList.toggle('open');
}

// ─── Token Visibility ───
function toggleTokenVisibility() {
    const input = apiTokenInput;
    const eyeIcon = document.getElementById('icon-eye');
    const eyeOffIcon = document.getElementById('icon-eye-off');

    if (input.type === 'password') {
        input.type = 'text';
        eyeIcon.style.display = 'none';
        eyeOffIcon.style.display = 'block';
    } else {
        input.type = 'password';
        eyeIcon.style.display = 'block';
        eyeOffIcon.style.display = 'none';
    }
}

// ─── Save Token ───
function saveToken() {
    const token = apiTokenInput.value.trim();
    if (!token) {
        showApiStatus('Please enter a valid token.', 'error');
        return;
    }
    if (!token.startsWith('hf_')) {
        showApiStatus('Token should start with "hf_"', 'error');
        return;
    }
    localStorage.setItem(TOKEN_KEY, token);
    showApiStatus('✓ Token saved successfully!', 'success');
    apiDot.classList.add('connected');
    setTimeout(() => toggleApiPanel(), 1200);
}

function loadTokenStatus() {
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) {
        apiDot.classList.add('connected');
        apiTokenInput.value = token;
    }
}

function showApiStatus(msg, type) {
    apiStatus.textContent = msg;
    apiStatus.className = `api-status-msg ${type}`;
    setTimeout(() => {
        apiStatus.textContent = '';
        apiStatus.className = 'api-status-msg';
    }, 4000);
}

function getToken() {
    return localStorage.getItem(TOKEN_KEY) || '';
}

// ─── Tab Switching ───
function showAnalyzeTab() {
    document.getElementById('tab-analyze').classList.add('active');
    document.getElementById('tab-examples').classList.remove('active');
    document.getElementById('input-area').style.display = 'block';
    document.getElementById('examples-panel').style.display = 'none';
}

function showExamplesTab() {
    document.getElementById('tab-examples').classList.add('active');
    document.getElementById('tab-analyze').classList.remove('active');
    document.getElementById('input-area').style.display = 'none';
    document.getElementById('examples-panel').style.display = 'block';
}

// ─── Load Example ───
function loadExample(btn) {
    const text = btn.getAttribute('data-text');
    textInput.value = text;
    updateCharCount();
    showAnalyzeTab();
    // Auto-analyze
    setTimeout(() => analyzeSentiment(), 300);
}

// ─── Clear Input ───
function clearInput() {
    textInput.value = '';
    updateCharCount();
    textInput.focus();
}

// ─── Analyze Sentiment ───
async function analyzeSentiment() {
    const text = textInput.value.trim();

    if (!text) {
        showToast('Please enter some text to analyze.', 'error');
        textInput.focus();
        return;
    }

    if (text.length < 3) {
        showToast('Please enter at least 3 characters.', 'error');
        return;
    }

    const token = getToken();
    if (!token) {
        showToast('Please add your Hugging Face API token first.', 'info');
        toggleApiPanel();
        return;
    }

    if (isAnalyzing) return;
    isAnalyzing = true;

    // UI: loading state
    btnAnalyze.classList.add('loading');
    btnAnalyze.disabled = true;

    const startTime = performance.now();

    try {
        const response = await fetch(HF_API_URL, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ inputs: text }),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            if (response.status === 401) {
                throw new Error('Invalid API token. Please check your Hugging Face token.');
            } else if (response.status === 503) {
                // Model is loading
                const estimatedTime = errData.estimated_time || 20;
                showToast(`Model is loading... Please wait ~${Math.ceil(estimatedTime)}s and try again.`, 'info');
                return;
            } else {
                throw new Error(errData.error || `API error (${response.status})`);
            }
        }

        const data = await response.json();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

        // Process results
        processResults(data, text, elapsed);

    } catch (error) {
        console.error('Analysis error:', error);
        showToast(error.message || 'Failed to analyze. Please try again.', 'error');
    } finally {
        isAnalyzing = false;
        btnAnalyze.classList.remove('loading');
        btnAnalyze.disabled = false;
    }
}

// ─── Process Results ───
function processResults(data, text, elapsed) {
    // HF returns [[{label, score}, ...]]
    let results = Array.isArray(data[0]) ? data[0] : data;

    // Normalize labels
    let positiveScore = 0;
    let negativeScore = 0;

    results.forEach(item => {
        const label = item.label.toUpperCase();
        if (label === 'POSITIVE' || label === 'LABEL_1' || label === 'POS') {
            positiveScore = item.score;
        } else if (label === 'NEGATIVE' || label === 'LABEL_0' || label === 'NEG') {
            negativeScore = item.score;
        }
    });

    // If only one score returned, calculate the other
    if (positiveScore === 0 && negativeScore === 0) {
        // Fallback: use the first result
        if (results.length > 0) {
            positiveScore = results[0].score;
            negativeScore = 1 - positiveScore;
        }
    } else if (positiveScore === 0) {
        positiveScore = 1 - negativeScore;
    } else if (negativeScore === 0) {
        negativeScore = 1 - positiveScore;
    }

    const isPositive = positiveScore >= negativeScore;
    const confidence = Math.max(positiveScore, negativeScore);

    // Display results
    displayResults({
        isPositive,
        positiveScore,
        negativeScore,
        confidence,
        text,
        elapsed,
    });

    // Save to history
    addToHistory({
        text,
        isPositive,
        confidence,
        timestamp: Date.now(),
    });
}

// ─── Display Results ───
function displayResults({ isPositive, positiveScore, negativeScore, confidence, text, elapsed }) {
    // Show section
    resultsSection.style.display = 'block';

    // Scroll into view
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    // Verdict
    const verdictEmoji = document.getElementById('verdict-emoji');
    const verdictLabel = document.getElementById('verdict-label');
    const verdictConfidence = document.getElementById('verdict-confidence');

    verdictEmoji.textContent = isPositive ? '😊' : '😞';
    verdictEmoji.style.animation = 'none';
    void verdictEmoji.offsetHeight; // Force reflow
    verdictEmoji.style.animation = 'verdictPop 0.5s cubic-bezier(0.68, -0.55, 0.27, 1.55)';

    verdictLabel.textContent = isPositive ? 'Positive' : 'Negative';
    verdictLabel.className = `verdict-label ${isPositive ? 'positive' : 'negative'}`;

    verdictConfidence.textContent = `${(confidence * 100).toFixed(1)}% confidence`;

    // Bars
    const positiveBar = document.getElementById('positive-bar');
    const negativeBar = document.getElementById('negative-bar');
    const positiveValue = document.getElementById('positive-value');
    const negativeValue = document.getElementById('negative-value');

    // Reset bars first
    positiveBar.style.width = '0%';
    negativeBar.style.width = '0%';

    // Animate bars
    requestAnimationFrame(() => {
        setTimeout(() => {
            positiveBar.style.width = `${(positiveScore * 100).toFixed(1)}%`;
            negativeBar.style.width = `${(negativeScore * 100).toFixed(1)}%`;
        }, 100);
    });

    positiveValue.textContent = `${(positiveScore * 100).toFixed(1)}%`;
    negativeValue.textContent = `${(negativeScore * 100).toFixed(1)}%`;

    // Time
    document.getElementById('results-time').textContent = `${elapsed}s`;

    // Analyzed text
    document.getElementById('analyzed-text').textContent = text;

    // Re-trigger animation
    resultsSection.style.animation = 'none';
    void resultsSection.offsetHeight;
    resultsSection.style.animation = 'slideUp 0.5s ease-out';
}

// ─── History Management ───
function loadHistory() {
    try {
        const stored = localStorage.getItem(HISTORY_KEY);
        history = stored ? JSON.parse(stored) : [];
        renderHistory();
    } catch {
        history = [];
    }
}

function addToHistory(entry) {
    history.unshift(entry);
    if (history.length > 10) history = history.slice(0, 10);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    renderHistory();
}

function clearHistory() {
    history = [];
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    showToast('History cleared.', 'success');
}

function renderHistory() {
    if (history.length === 0) {
        historySection.style.display = 'none';
        return;
    }

    historySection.style.display = 'block';
    historyGrid.innerHTML = '';

    history.forEach((entry, index) => {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.setAttribute('tabindex', '0');
        item.setAttribute('role', 'button');
        item.setAttribute('aria-label', `Re-analyze: ${entry.text.substring(0, 50)}`);

        const truncatedText = entry.text.length > 80 
            ? entry.text.substring(0, 80) + '...' 
            : entry.text;

        const timeAgo = getTimeAgo(entry.timestamp);

        item.innerHTML = `
            <span class="history-emoji">${entry.isPositive ? '😊' : '😞'}</span>
            <div class="history-content">
                <div class="history-text">${escapeHtml(truncatedText)}</div>
                <div class="history-meta">
                    <span class="history-label ${entry.isPositive ? 'positive' : 'negative'}">
                        ${entry.isPositive ? 'Positive' : 'Negative'}
                    </span>
                    <span class="history-score">${(entry.confidence * 100).toFixed(1)}% · ${timeAgo}</span>
                </div>
            </div>
        `;

        item.addEventListener('click', () => {
            textInput.value = entry.text;
            updateCharCount();
            showAnalyzeTab();
            textInput.scrollIntoView({ behavior: 'smooth' });
        });

        historyGrid.appendChild(item);
    });
}

// ─── Utilities ───
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getTimeAgo(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'just now';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

// ─── Toast Notification ───
let activeToast = null;

function showToast(message, type = 'error') {
    // Remove existing toast
    if (activeToast) {
        activeToast.remove();
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        error: '⚠️',
        success: '✅',
        info: 'ℹ️',
    };

    toast.innerHTML = `<span>${icons[type] || ''}</span><span>${escapeHtml(message)}</span>`;
    document.body.appendChild(toast);
    activeToast = toast;

    // Trigger animation
    requestAnimationFrame(() => {
        toast.classList.add('show');
    });

    // Auto dismiss
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toast.remove();
            if (activeToast === toast) activeToast = null;
        }, 400);
    }, 4500);
}
