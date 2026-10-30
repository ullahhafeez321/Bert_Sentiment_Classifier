const MODEL_NAME = 'ullahhafeez321/bert-sentiment-classifier';
const textInput = document.getElementById('reviewText');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const emptyState = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
const loadingState = document.getElementById('loadingState');

textInput.addEventListener('input', () => {
  document.getElementById('charCount').textContent = `${textInput.value.length.toLocaleString()} / 5,000 characters`;
  errorMessage.hidden = true;
});
document.getElementById('clearBtn').addEventListener('click', () => {
  textInput.value = '';
  textInput.dispatchEvent(new Event('input'));
  textInput.focus();
});
document.querySelectorAll('.example').forEach(button => button.addEventListener('click', () => {
  textInput.value = button.dataset.text;
  textInput.dispatchEvent(new Event('input'));
  textInput.focus();
}));
document.getElementById('year').textContent = new Date().getFullYear();

function normalizeScores(payload) {
  // Hugging Face text-classification endpoints commonly return either
  // [{label, score}, ...] or [[{label, score}, ...]].
  let rows = payload;
  if (Array.isArray(rows) && rows.length === 1 && Array.isArray(rows[0])) rows = rows[0];
  if (Array.isArray(rows) && rows.length && rows[0] && Array.isArray(rows[0])) rows = rows[0];
  if (Array.isArray(rows)) return rows.filter(x => x && typeof x.label === 'string' && Number.isFinite(Number(x.score)));
  if (rows && typeof rows === 'object' && typeof rows.label === 'string' && Number.isFinite(Number(rows.score))) return [rows];
  throw new Error('The model returned an unexpected response. Check your Hugging Face endpoint configuration.');
}
function sentimentType(label) {
  const value = label.toLowerCase();
  if (/positive|pos|good|favorable|favourable/.test(value)) return 'positive';
  if (/negative|neg|bad|unfavorable|unfavourable/.test(value)) return 'negative';
  if (/neutral/.test(value)) return 'neutral';
  return 'other';
}
function prettyLabel(label) {
  return label.replace(/[_-]+/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
function showResults(scores, originalText) {
  scores.sort((a,b) => Number(b.score) - Number(a.score));
  const best = scores[0];
  const type = sentimentType(best.label);
  const percent = Math.max(0, Math.min(100, Number(best.score) * 100));
  const icon = document.getElementById('sentimentIcon');
  icon.className = `sentiment-icon ${type}`;
  icon.textContent = type === 'positive' ? '✦' : type === 'negative' ? '⌁' : '◈';
  document.getElementById('sentimentLabel').textContent = prettyLabel(best.label);
  document.getElementById('sentimentSubtext').textContent = 'Highest scoring class from your BERT model';
  document.getElementById('confidenceValue').textContent = `${percent.toFixed(1)}%`;
  document.getElementById('confidenceBar').style.width = `${percent}%`;
  document.getElementById('confidenceBar').style.background = type === 'negative' ? 'var(--red)' : type === 'neutral' ? 'var(--blue)' : 'var(--mint)';
  document.getElementById('allScores').innerHTML = scores.map(item => {
    const value = Math.max(0, Math.min(100, Number(item.score) * 100));
    return `<div class="score-row"><span>${escapeHtml(prettyLabel(item.label))}</span><b>${value.toFixed(1)}%</b><div class="mini-track"><i style="width:${value}%"></i></div></div>`;
  }).join('');
  document.getElementById('analyzedText').textContent = originalText;
  emptyState.hidden = true;
  resultContent.hidden = false;
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}
async function analyze() {
  const text = textInput.value.trim();
  errorMessage.hidden = true;
  if (!text) {
    errorMessage.textContent = 'Please enter some text before analyzing.';
    errorMessage.hidden = false;
    textInput.focus();
    return;
  }
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector('.button-text').textContent = 'Analyzing...';
  emptyState.hidden = true;
  resultContent.hidden = true;
  loadingState.hidden = false;
  try {
    const response = await fetch('/.netlify/functions/predict', {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text})
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    const scores = normalizeScores(data);
    if (!scores.length) throw new Error('No class scores were returned by the model.');
    showResults(scores, text);
  } catch (error) {
    emptyState.hidden = false;
    errorMessage.textContent = error.message || 'Something went wrong while contacting the model.';
    errorMessage.hidden = false;
  } finally {
    loadingState.hidden = true;
    analyzeBtn.disabled = false;
    analyzeBtn.querySelector('.button-text').textContent = 'Analyze sentiment';
  }
}
analyzeBtn.addEventListener('click', analyze);
textInput.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') analyze(); });
