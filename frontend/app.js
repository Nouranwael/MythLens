const tabs = [...document.querySelectorAll('.tab')];
const panels = {
  text: document.getElementById('panel-text'),
  url: document.getElementById('panel-url'),
  video: document.getElementById('panel-video'),
};
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingCard = document.getElementById('loadingCard');
const errorCard = document.getElementById('errorCard');
const resultSection = document.getElementById('resultSection');
const claimsContainer = document.getElementById('claimsContainer');
const summaryCard = document.getElementById('summaryCard');
const summaryText = document.getElementById('summaryText');
const languageBadge = document.getElementById('languageBadge');
let mode = 'text';

tabs.forEach((tab) => tab.addEventListener('click', () => {
  tabs.forEach((item) => item.classList.remove('active'));
  tab.classList.add('active');
  mode = tab.dataset.mode;
  Object.entries(panels).forEach(([key, panel]) => panel.classList.toggle('active', key === mode));
}));

function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;',
  }[character]));
}

function evidenceMarkup(items = []) {
  if (!items.length) return '<p class="helper">No evidence returned.</p>';
  return `<div class="evidence-list">${items.slice(0, 5).map((item) => `
    <article class="evidence-item">
      <div class="evidence-title">${escapeHtml(item.title || 'Untitled source')}</div>
      <div class="evidence-meta">${escapeHtml(item.source || '')} ${item.study_type ? `- ${escapeHtml(item.study_type)}` : ''} ${item.score !== undefined ? `- Score ${Number(item.score).toFixed(3)}` : ''}</div>
      ${item.url ? `<div class="evidence-meta"><a href="${escapeHtml(item.url)}" target="_blank" rel="noopener">Open source</a>${item.pmid ? ` - PMID ${escapeHtml(item.pmid)}` : ''}</div>` : ''}
    </article>`).join('')}</div>`;
}

function claimCard(claim, index, result = null) {
  const verified = Boolean(result);
  const confidence = Math.round((Number(result?.confidence) || 0) * 100);
  return `<article class="card claim-card">
    <div class="claim-top">
      <div>
        <div class="label">Claim ${index + 1}</div>
        <p class="claim-text">${escapeHtml(claim.normalized_claim || claim.original_claim || '')}</p>
      </div>
      ${verified ? `<div class="badges"><span class="badge verdict">${escapeHtml(result.verdict || 'UNPROVEN')}</span><span class="badge">Risk: ${escapeHtml(result.risk_level || 'LOW')}</span></div>` : '<span class="badge neutral">Claim extracted</span>'}
    </div>
    <div class="divider"></div>
    <div class="grid-two">
      <div><span class="label">Medical query:</span><p class="query">${escapeHtml(claim.medical_query || '')}</p></div>
    </div>
    ${verified ? `<div class="divider"></div><div class="grid-two"><div><span class="label">Confidence:</span><strong>${confidence}%</strong></div><div><span class="label">Evidence:</span><strong>${result.insufficient_evidence ? 'Insufficient' : 'Available'}</strong></div></div><div class="divider"></div><div><div class="label">Explanation:</div><p class="value">${escapeHtml(result.explanation_ar || '')}</p></div><div class="divider"></div><div><div class="label">Safe recommendation:</div><p class="value">${escapeHtml(result.safe_recommendation || '')}</p></div><div class="divider"></div>${evidenceMarkup(result.evidence || result.citations || [])}` : '<p class="helper">Verification is running. The Groq-extracted claim is already available.</p>'}
  </article>`;
}

function renderPrepared(data) {
  languageBadge.textContent = data.language || '';
  summaryText.textContent = data.summary || '';
  summaryCard.classList.toggle('hidden', !data.summary);
  const claims = data.claims || [];
  claimsContainer.innerHTML = claims.length
    ? claims.map((claim, index) => claimCard(claim, index)).join('')
    : '<div class="card claim-card"><p>No health claims were returned by Groq.</p></div>';
  resultSection.classList.remove('hidden');
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderVerified(data) {
  const claims = data.claims || [];
  const results = data.results || [];
  claimsContainer.innerHTML = claims.map((claim, index) => claimCard(claim, index, results[index] || null)).join('');
}

function setLoading(isLoading, message = '') {
  loadingCard.classList.toggle('hidden', !isLoading);
  analyzeBtn.disabled = isLoading;
  analyzeBtn.textContent = isLoading ? (message || 'Analyzing...') : 'Analyze';
}

function showError(message) {
  errorCard.textContent = message;
  errorCard.classList.remove('hidden');
}

async function fetchPrepared() {
  if (mode === 'text') {
    const text = document.getElementById('claimText').value.trim();
    if (text.length < 3) throw new Error('Enter a medical claim first.');
    return fetch('/api/prepare/text', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, top_k: 3 }) });
  }
  if (mode === 'url') {
    const url = document.getElementById('videoUrl').value.trim();
    if (!url) throw new Error('Enter a video URL first.');
    return fetch('/api/prepare/url', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url, top_k: 3 }) });
  }
  const file = document.getElementById('videoFile').files[0];
  if (!file) throw new Error('Choose a video or audio file first.');
  const form = new FormData();
  form.append('file', file);
  form.append('top_k', '3');
  return fetch('/api/prepare/video', { method: 'POST', body: form });
}

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'The request failed.');
  return data;
}

async function requestAnalysis() {
  errorCard.classList.add('hidden');
  resultSection.classList.add('hidden');
  setLoading(true, 'Extracting claims...');
  try {
    const prepared = await readResponse(await fetchPrepared());
    renderPrepared(prepared);
    if (!prepared.claims?.length) return;
    setLoading(true, 'Claims ready - retrieving evidence...');
    const verified = await readResponse(await fetch('/api/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...prepared, top_k: 3 }),
    }));
    renderVerified(verified);
  } catch (error) {
    showError(error.message || 'Unexpected analysis error.');
  } finally {
    setLoading(false);
  }
}

analyzeBtn.addEventListener('click', requestAnalysis);
