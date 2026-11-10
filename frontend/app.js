/* ═══════════════════════════════════════════════════
   SecureRAG Frontend — app.js
   ═══════════════════════════════════════════════════ */

const API = '/api';
const INGEST_KEY = 'securerag-ingest-2024';

let sessionId = generateUUID();
let isLoading = false;
let evalChart = null;
let ingestPollTimer = null;
let chatHistory = [];

// ─── Utilities ─────────────────────────────────────

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function severityClass(sev) {
  const s = (sev || '').toUpperCase();
  if (s === 'CRITICAL') return 'badge-critical';
  if (s === 'HIGH') return 'badge-high';
  if (s === 'MEDIUM') return 'badge-medium';
  if (s === 'LOW') return 'badge-low';
  return 'badge-medium';
}

function scoreClass(val) {
  if (val >= 0.8) return 'score-good';
  if (val >= 0.5) return 'score-ok';
  return 'score-bad';
}

function scoreLabel(val) {
  if (val >= 0.8) return ['Excellent', 'label-excellent'];
  if (val >= 0.6) return ['Good', 'label-good'];
  return ['Needs Improvement', 'label-needs'];
}

function scoreColor(val) {
  if (val >= 0.8) return 'var(--accent-green)';
  if (val >= 0.5) return 'var(--accent-yellow)';
  return 'var(--accent-red)';
}

// ─── DOM Elements ──────────────────────────────────

const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeScreen = document.getElementById('welcomeScreen');
const viewChat = document.getElementById('viewChat');
const viewEval = document.getElementById('viewEval');
const tabChat = document.getElementById('tabChat');
const tabEval = document.getElementById('tabEval');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const ingestBtn = document.getElementById('ingestBtn');
const ingestBtnText = document.getElementById('ingestBtnText');
const ingestProgress = document.getElementById('ingestProgress');
const ingestFill = document.getElementById('ingestFill');
const ingestLabel = document.getElementById('ingestLabel');
const newChatBtn = document.getElementById('newChatBtn');
const runEvalBtn = document.getElementById('runEvalBtn');
const evalEmpty = document.getElementById('evalEmpty');
const evalResults = document.getElementById('evalResults');
const evalRunning = document.getElementById('evalRunning');
const evalTimestamp = document.getElementById('evalTimestamp');
const sourceModal = document.getElementById('sourceModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const modalClose = document.getElementById('modalClose');
const hamburgerBtn = document.getElementById('hamburgerBtn');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const exportBtn = document.getElementById('exportBtn');

// Stats elements
const statDocs = document.getElementById('statDocs');
const statCVEs = document.getElementById('statCVEs');
const statOWASP = document.getElementById('statOWASP');
const statQueries = document.getElementById('statQueries');

// ─── Tab switching ──────────────────────────────────

function switchTab(tab) {
  const isChat = tab === 'chat';
  tabChat.classList.toggle('active', isChat);
  tabEval.classList.toggle('active', !isChat);
  viewChat.classList.toggle('active', isChat);
  viewEval.classList.toggle('active', !isChat);
}

tabChat.addEventListener('click', () => switchTab('chat'));
tabEval.addEventListener('click', () => switchTab('eval'));

// ─── Sidebar / Mobile ──────────────────────────────

hamburgerBtn.addEventListener('click', () => {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('open');
});

sidebarOverlay.addEventListener('click', () => {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('open');
});

// ─── Auto-resize textarea ──────────────────────────

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

// ─── Example queries ───────────────────────────────

document.querySelectorAll('.example-query').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.query;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// ─── New Chat ──────────────────────────────────────

newChatBtn.addEventListener('click', () => {
  // Clear session on server
  fetch(`${API}/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
  sessionId = generateUUID();
  chatHistory = [];

  // Clear messages but keep welcome screen
  const msgs = messagesContainer.querySelectorAll('.message');
  msgs.forEach(m => m.remove());
  if (welcomeScreen) welcomeScreen.style.display = '';
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('open');
});

// ─── Filter helpers ────────────────────────────────

function getActiveFilters() {
  const severityFilter = Array.from(document.querySelectorAll('.sev-filter:checked'))
    .map(cb => cb.value);
  const sourceFilterEl = document.querySelector('.src-filter:checked');
  const sourceFilter = sourceFilterEl ? sourceFilterEl.value : 'both';
  return { severityFilter, sourceFilter };
}

// ─── Send message ──────────────────────────────────

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isLoading) return;

  if (welcomeScreen) welcomeScreen.style.display = 'none';

  chatInput.value = '';
  chatInput.style.height = 'auto';
  isLoading = true;
  sendBtn.disabled = true;

  appendMessage('user', text);
  const typingEl = appendTypingIndicator();

  const { severityFilter, sourceFilter } = getActiveFilters();

  try {
    // use the streaming endpoint — tokens come in as they're generated by the LLM
    // instead of waiting for the whole response (which could take 3-5s)
    const resp = await fetch(`${API}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: sessionId,
        severity_filter: severityFilter,
        source_filter: sourceFilter,
      }),
    });

    typingEl.remove();

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Request failed' }));
      appendMessage('assistant', `**Error:** ${err.detail || 'Something went wrong. Please try again.'}`, [], {}, 0);
      return;
    }

    // build the message element early so we can append tokens into it live
    const msgEl = buildMessageEl('assistant', '', [], {}, 0);
    messagesContainer.appendChild(msgEl);
    const bubbleEl = msgEl.querySelector('.message-bubble');
    const contentEl = msgEl.querySelector('.message-content');

    let fullText = '';
    let sources = [];
    let searchMeta = {};
    let responseTimeMs = 0;

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE lines are separated by \n\n
      const lines = buffer.split('\n');
      buffer = lines.pop();  // keep the incomplete line for next chunk

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try {
          evt = JSON.parse(line.slice(6));
        } catch {
          continue;
        }

        if (evt.type === 'metadata') {
          sources = evt.sources || [];
          searchMeta = evt.search_metadata || {};
          // render source chips right away — user can see what's being referenced
          // while the answer is still coming in
          if (sources.length > 0) {
            const srcEl = buildSourcesEl(sources);
            contentEl.insertBefore(srcEl, contentEl.firstChild);
          }
        } else if (evt.type === 'token') {
          fullText += evt.content;
          bubbleEl.innerHTML = marked.parse(fullText);
          scrollToBottom();
        } else if (evt.type === 'verification') {
          // IDs the answer cited that weren't in the retrieved context — flag, don't hide
          if (Array.isArray(evt.unverified) && evt.unverified.length > 0) {
            const warnEl = document.createElement('div');
            warnEl.className = 'citation-warning';
            warnEl.style.cssText = 'margin-top:8px;padding:6px 10px;border-radius:6px;font-size:12px;' +
              'background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.4);color:#f59e0b;';
            warnEl.title = 'These IDs appear in the answer but not in the retrieved sources';
            warnEl.textContent = '⚠ Unverified citation' + (evt.unverified.length > 1 ? 's' : '') +
              ': ' + evt.unverified.join(', ') + ' — not found in retrieved sources, double-check.';
            contentEl.appendChild(warnEl);
          }
        } else if (evt.type === 'done') {
          responseTimeMs = evt.response_time_ms || 0;
        } else if (evt.type === 'error') {
          bubbleEl.innerHTML = marked.parse(`**Stream error:** ${evt.message}`);
        }
      }
    }

    // finalize
    sessionId = resp.headers.get('X-Session-ID') || sessionId;
    chatHistory.push({ role: 'user', content: text });
    chatHistory.push({ role: 'assistant', content: fullText });

    // add timing metadata footer
    if (responseTimeMs > 0) {
      const metaEl = document.createElement('div');
      metaEl.className = 'message-meta';
      const method = searchMeta.bm25_hits ? 'Hybrid (BM25 + Vector → RRF → Reranked)' : 'Vector';
      const filteredPart = searchMeta.filtered_out > 0
        ? `<span>·</span><span title="Excluded by active filters">${searchMeta.filtered_out} filtered</span>`
        : '';
      metaEl.innerHTML = `<span>${responseTimeMs}ms</span><span>·</span><span>${method}</span>${filteredPart}`;
      contentEl.appendChild(metaEl);
    }

    // follow-up suggestions
    const suggestions = generateFollowUps(fullText);
    if (suggestions.length > 0) {
      const followupEl = document.createElement('div');
      followupEl.className = 'followup-suggestions';
      suggestions.forEach(s => {
        const btn = document.createElement('button');
        btn.className = 'followup-btn';
        btn.textContent = s;
        btn.addEventListener('click', () => {
          chatInput.value = s;
          chatInput.dispatchEvent(new Event('input'));
          sendMessage();
        });
        followupEl.appendChild(btn);
      });
      contentEl.appendChild(followupEl);
    }

    scrollToBottom();
  } catch (e) {
    typingEl.remove();
    appendMessage('assistant', '**Connection error.** Please check that the server is running and try again.', [], {}, 0);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    fetchStats();
  }
}

// ─── Message rendering ─────────────────────────────

function appendMessage(role, content, sources = [], meta = {}, responseTime = 0) {
  const el = buildMessageEl(role, content, sources, meta, responseTime);
  messagesContainer.appendChild(el);
  scrollToBottom();
  return el;
}

function buildMessageEl(role, content, sources = [], meta = {}, responseTime = 0) {
  const el = document.createElement('div');
  el.className = `message ${role}`;

  const avatarEl = document.createElement('div');
  avatarEl.className = 'message-avatar';
  avatarEl.textContent = role === 'user' ? 'U' : 'S';

  const contentEl = document.createElement('div');
  contentEl.className = 'message-content';

  const bubbleEl = document.createElement('div');
  bubbleEl.className = 'message-bubble';
  if (content) bubbleEl.innerHTML = marked.parse(content);

  contentEl.appendChild(bubbleEl);

  if (role === 'assistant' && sources && sources.length > 0) {
    const sourcesEl = buildSourcesEl(sources);
    contentEl.appendChild(sourcesEl);
  }

  if (role === 'assistant' && responseTime > 0) {
    const metaEl = document.createElement('div');
    metaEl.className = 'message-meta';
    const searchMethod = meta.bm25_hits ? `Hybrid (BM25 + Vector → RRF → Reranked)` : 'Vector';
    const filteredPart = meta.filtered_out > 0
      ? `<span>·</span><span title="Documents excluded by active filters">${meta.filtered_out} filtered</span>`
      : '';
    metaEl.innerHTML = `<span>${responseTime}ms</span><span>·</span><span>${searchMethod}</span>${filteredPart}`;
    contentEl.appendChild(metaEl);
  }

  if (role === 'user') {
    el.appendChild(contentEl);
    el.appendChild(avatarEl);
  } else {
    el.appendChild(avatarEl);
    el.appendChild(contentEl);
  }

  return el;
}

function buildSourcesEl(sources) {
  const el = document.createElement('div');
  el.className = 'message-sources';

  sources.forEach(src => {
    const chip = document.createElement('button');
    chip.className = `source-chip ${src.source_type === 'NVD' ? 'nvd' : 'owasp'}`;

    if (src.source_type === 'NVD') {
      const sev = (src.severity || 'UNKNOWN').toUpperCase();
      chip.innerHTML = `${escapeHTML(src.id)} <span class="badge ${severityClass(sev)}">${sev}</span>`;
    } else {
      chip.innerHTML = `${escapeHTML(src.id)}`;
    }

    chip.addEventListener('click', () => openSourceModal(src));
    el.appendChild(chip);
  });

  return el;
}

function appendTypingIndicator() {
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="message-avatar">S</div>
    <div class="message-content">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  messagesContainer.appendChild(el);
  scrollToBottom();
  return el;
}

function scrollToBottom() {
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// ─── Follow-up suggestions ─────────────────────────

function generateFollowUps(answerText) {
  const suggestions = [];
  const lower = answerText.toLowerCase();

  if (lower.includes('sql injection') || lower.includes('a03')) {
    suggestions.push('How do parameterized queries prevent SQL injection?');
  }
  if (lower.includes('access control') || lower.includes('a01')) {
    suggestions.push('What are common access control bypass techniques?');
  }
  if (lower.includes('ssrf') || lower.includes('a10')) {
    suggestions.push('How can SSRF be used to access cloud metadata?');
  }
  if (lower.includes('authentication') || lower.includes('a07')) {
    suggestions.push('What is multi-factor authentication and how does it help?');
  }
  if (lower.includes('cryptograph') || lower.includes('encryption')) {
    suggestions.push('What are the recommended encryption algorithms for data at rest?');
  }
  if (lower.includes('cve')) {
    suggestions.push('What is the CVSS scoring system?');
  }
  if (lower.includes('owasp')) {
    suggestions.push('How often does the OWASP Top 10 get updated?');
  }

  return suggestions.slice(0, 2);
}

// ─── Source modal ──────────────────────────────────

function openSourceModal(src) {
  modalTitle.textContent = src.id;

  const rows = [];
  rows.push({ key: 'Source Type', value: src.source_type });
  if (src.severity) rows.push({ key: 'Severity', value: `<span class="badge ${severityClass(src.severity)}">${src.severity}</span>` });
  if (src.score) rows.push({ key: 'CVSS Score', value: src.score.toString() });
  if (src.category) rows.push({ key: 'Category', value: src.category });

  modalBody.innerHTML = rows.map(r => `
    <div class="modal-row">
      <span class="modal-row-key">${r.key}</span>
      <span class="modal-row-value">${r.value}</span>
    </div>
  `).join('');

  sourceModal.style.display = 'flex';
}

modalClose.addEventListener('click', () => { sourceModal.style.display = 'none'; });
sourceModal.addEventListener('click', e => {
  if (e.target === sourceModal) sourceModal.style.display = 'none';
});

// ─── Health check ──────────────────────────────────

async function fetchHealth() {
  try {
    const resp = await fetch(`${API}/health`);
    if (!resp.ok) throw new Error('not ok');
    const data = await resp.json();

    if (data.chromadb_documents === 0) {
      setStatus('ready', 'Empty KB');
    } else {
      setStatus('ready', data.models_loaded ? 'Ready' : 'Loading...');
    }
  } catch {
    setStatus('error', 'Offline');
  }
}

function setStatus(state, text) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text;
}

// ─── Stats ─────────────────────────────────────────

async function fetchStats() {
  try {
    const resp = await fetch(`${API}/stats`);
    if (!resp.ok) return;
    const d = await resp.json();
    statDocs.textContent = d.total_documents.toLocaleString();
    statCVEs.textContent = d.cve_count.toLocaleString();
    statOWASP.textContent = d.owasp_count.toLocaleString();
    statQueries.textContent = d.queries_served.toLocaleString();
  } catch {}
}

// ─── Ingest ────────────────────────────────────────

ingestBtn.addEventListener('click', async () => {
  if (ingestBtn.disabled) return;
  ingestBtn.disabled = true;
  ingestBtnText.textContent = 'Starting...';
  setStatus('ingesting', 'Ingesting...');

  try {
    const resp = await fetch(`${API}/ingest`, {
      method: 'POST',
      headers: { 'X-Ingest-Key': INGEST_KEY },
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      alert(err.detail || 'Ingest failed to start');
      ingestBtn.disabled = false;
      ingestBtnText.textContent = 'Ingest Data';
      setStatus('error', 'Error');
      return;
    }
    ingestProgress.style.display = 'block';
    startIngestPolling();
  } catch {
    alert('Failed to connect to server');
    ingestBtn.disabled = false;
    ingestBtnText.textContent = 'Ingest Data';
  }
});

function startIngestPolling() {
  if (ingestPollTimer) clearInterval(ingestPollTimer);
  ingestPollTimer = setInterval(pollIngestStatus, 3000);
}

async function pollIngestStatus() {
  try {
    const resp = await fetch(`${API}/ingest/status`);
    if (!resp.ok) return;
    const d = await resp.json();

    if (d.total > 0) {
      const pct = Math.round((d.progress / d.total) * 100);
      ingestFill.style.width = pct + '%';
      ingestLabel.textContent = `${d.progress.toLocaleString()} / ${d.total.toLocaleString()}`;
      ingestBtnText.textContent = `Ingesting ${pct}%`;
    }

    if (d.status === 'completed') {
      clearInterval(ingestPollTimer);
      ingestPollTimer = null;
      ingestBtn.disabled = false;
      ingestBtnText.textContent = 'Ingest Data';
      ingestFill.style.width = '100%';
      ingestLabel.textContent = 'Complete!';
      setStatus('ready', 'Ready');
      fetchStats();
      setTimeout(() => { ingestProgress.style.display = 'none'; }, 3000);
    } else if (d.status === 'failed') {
      clearInterval(ingestPollTimer);
      ingestPollTimer = null;
      ingestBtn.disabled = false;
      ingestBtnText.textContent = 'Ingest Data';
      ingestLabel.textContent = 'Failed!';
      setStatus('error', 'Ingest failed');
    }
  } catch {}
}

// ─── Evaluation Dashboard ──────────────────────────

runEvalBtn.addEventListener('click', async () => {
  if (runEvalBtn.disabled) return;
  runEvalBtn.disabled = true;
  evalEmpty.style.display = 'none';
  evalResults.style.display = 'none';
  evalRunning.style.display = 'block';
  evalTimestamp.textContent = '';

  try {
    const resp = await fetch(`${API}/evaluate`, { method: 'POST' });
    evalRunning.style.display = 'none';

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      evalEmpty.style.display = 'block';
      alert(err.detail || 'Evaluation failed. Make sure the knowledge base is populated.');
    } else {
      const data = await resp.json();
      renderEvalResults(data);
    }
  } catch {
    evalRunning.style.display = 'none';
    evalEmpty.style.display = 'block';
    alert('Evaluation failed — server error');
  } finally {
    runEvalBtn.disabled = false;
  }
});

function renderEvalResults(data) {
  evalResults.style.display = 'block';
  evalTimestamp.textContent = `Last run: ${new Date(data.evaluated_at).toLocaleString()}`;

  const scores = data.scores || {};
  const metricsOrder = [
    ['faithfulness', 'Faithfulness'],
    ['answer_relevancy', 'Answer Relevancy'],
    ['context_recall', 'Context Recall'],
    ['context_precision', 'Context Precision'],
  ];

  // Metric cards
  const cardsEl = document.getElementById('metricCards');
  cardsEl.innerHTML = '';
  metricsOrder.forEach(([key, label]) => {
    const val = scores[key] || 0;
    const [lbl, lblClass] = scoreLabel(val);
    const color = scoreColor(val);
    const card = document.createElement('div');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-card-name">${label}</div>
      <div class="metric-card-score" style="color:${color}">${val.toFixed(2)}</div>
      <div class="metric-card-bar">
        <div class="metric-card-fill" style="width:0%;background:${color}" data-width="${(val * 100).toFixed(1)}%"></div>
      </div>
      <div class="metric-card-label ${lblClass}">${lbl}</div>
    `;
    cardsEl.appendChild(card);
  });

  // Animate bars
  requestAnimationFrame(() => {
    document.querySelectorAll('.metric-card-fill').forEach(el => {
      el.style.width = el.dataset.width;
    });
  });

  // Chart
  renderEvalChart(metricsOrder, scores);

  // Table
  const tbody = document.getElementById('evalTableBody');
  tbody.innerHTML = '';
  (data.per_query_results || []).forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${escapeHTML(row.query)}</td>
      <td><span class="score-cell ${scoreClass(row.faithfulness)}">${(row.faithfulness || 0).toFixed(2)}</span></td>
      <td><span class="score-cell ${scoreClass(row.answer_relevancy)}">${(row.answer_relevancy || 0).toFixed(2)}</span></td>
      <td><span class="score-cell ${scoreClass(row.context_recall)}">${(row.context_recall || 0).toFixed(2)}</span></td>
      <td><span class="score-cell ${scoreClass(row.context_precision)}">${(row.context_precision || 0).toFixed(2)}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function renderEvalChart(metricsOrder, scores) {
  const ctx = document.getElementById('evalChart').getContext('2d');
  if (evalChart) evalChart.destroy();

  const labels = metricsOrder.map(([, label]) => label);
  const values = metricsOrder.map(([key]) => scores[key] || 0);
  const colors = values.map(scoreColor);

  evalChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Score',
        data: values,
        backgroundColor: colors.map(c => c.replace('var(--accent-green)', 'rgba(16,185,129,0.6)').replace('var(--accent-yellow)', 'rgba(234,179,8,0.6)').replace('var(--accent-red)', 'rgba(239,68,68,0.6)')),
        borderColor: colors.map(c => c.replace('var(--accent-green)', 'rgb(16,185,129)').replace('var(--accent-yellow)', 'rgb(234,179,8)').replace('var(--accent-red)', 'rgb(239,68,68)')),
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: {
            color: '#94a3b8',
            font: { family: "'JetBrains Mono', monospace", size: 11 },
          },
          grid: { color: 'rgba(255,255,255,0.05)' },
        },
        x: {
          ticks: { color: '#94a3b8', font: { family: "'Inter', sans-serif", size: 12 } },
          grid: { display: false },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f1525',
          borderColor: '#1e2d45',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          callbacks: {
            label: ctx => ` Score: ${ctx.raw.toFixed(3)}`,
          },
        },
      },
    },
  });
}

// ─── Export chat ───────────────────────────────────

exportBtn.addEventListener('click', () => {
  if (chatHistory.length === 0) {
    alert('No chat to export.');
    return;
  }
  let md = `# SecureRAG Chat Export\n_Exported: ${new Date().toLocaleString()}_\n\n---\n\n`;
  chatHistory.forEach(turn => {
    md += `**${turn.role === 'user' ? 'You' : 'SecureRAG'}:** ${turn.content}\n\n`;
  });
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `securerag-chat-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
});

// ─── Bootstrap ─────────────────────────────────────

fetchHealth();
fetchStats();
setInterval(fetchHealth, 30000);
setInterval(fetchStats, 60000);
