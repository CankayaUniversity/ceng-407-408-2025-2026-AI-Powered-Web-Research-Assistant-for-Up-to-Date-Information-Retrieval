const app = document.getElementById('app');
const conversation = document.getElementById('conversation');
const emptyState = document.getElementById('empty-state');
const form = document.getElementById('ask-form');
const input = document.getElementById('question-input');
const submitBtn = document.getElementById('submit-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const sidebarClose = document.getElementById('sidebar-close');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const chatList = document.getElementById('chat-list');
const sidebarEmpty = document.getElementById('sidebar-empty');
const searchInput = document.getElementById('search-input');
const searchClear = document.getElementById('search-clear');
const headerTitle = document.getElementById('header-title');
const deleteChatBtn = document.getElementById('delete-chat-btn');
const statusPill = document.getElementById('status-pill');
const statusText = statusPill.querySelector('.status-text');
const modelToggle = document.getElementById('model-toggle');

let isAsking = false;
let activeSource = null;
let activeChatId = null;
let cachedChats = [];
let searchDebounce = null;
let selectedModel = localStorage.getItem('selectedModel') || 'llama';
let activeTraceSpinner = null;
let activeTraceLabel = null;

const MODEL_LABELS = {
  llama: 'Llama 3.1',
  qwen: 'Qwen 2.5',
  llama32: 'Llama 3.2',
};

if (window.marked && marked.setOptions) {
  marked.setOptions({ breaks: true, gfm: true });
}

/* ----------------- send / stop button ----------------- */

const SEND_ICON = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>`;
const STOP_ICON = `<svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2.5"/></svg>`;

function setSubmitMode(mode) {
  if (mode === 'stop') {
    submitBtn.innerHTML = STOP_ICON;
    submitBtn.classList.add('stop-mode');
    submitBtn.disabled = false;
    submitBtn.title = 'Stop (Esc)';
  } else {
    submitBtn.innerHTML = SEND_ICON;
    submitBtn.classList.remove('stop-mode');
    submitBtn.disabled = false;
    submitBtn.title = 'Send (Enter)';
    activeTraceSpinner = null;
    activeTraceLabel = null;
  }
}

function stopResearch() {
  if (activeSource) {
    activeSource.close();
    activeSource = null;
  }
  isAsking = false;
  setStatus(null, 'Online');

  // Update trace BEFORE setSubmitMode clears the refs
  if (activeTraceSpinner) {
    const stopIcon = el(
      'span',
      'trace-stop-icon',
      '<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2.5"/></svg>'
    );
    activeTraceSpinner.replaceWith(stopIcon);
    activeTraceSpinner = null;
  }
  if (activeTraceLabel) {
    activeTraceLabel.textContent = 'Stopped';
    activeTraceLabel = null;
  }

  setSubmitMode('send');
}

// Intercept submit-button click while searching → act as stop button
submitBtn.addEventListener('click', (e) => {
  if (isAsking) {
    e.preventDefault();
    stopResearch();
  }
});

// Escape key shortcut
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isAsking) stopResearch();
});

/* ----------------- helpers ----------------- */

function el(tag, classes, html) {
  const node = document.createElement(tag);
  if (classes) node.className = classes;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function txt(tag, classes, content) {
  const node = document.createElement(tag);
  if (classes) node.className = classes;
  if (content !== undefined) node.textContent = content;
  return node;
}

function faviconUrl(domain) {
  return `https://www.google.com/s2/favicons?domain=${encodeURIComponent(domain)}&sz=64`;
}

function domainOf(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

function relativeTime(iso) {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (!then) return '';
  const diffSec = Math.max(0, (Date.now() - then) / 1000);
  if (diffSec < 60) return 'just now';
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 86400 * 7) return `${Math.floor(diffSec / 86400)}d ago`;
  return new Date(iso).toLocaleDateString();
}

function autosizeInput() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    conversation.scrollTop = conversation.scrollHeight;
  });
}

function setStatus(state, label) {
  statusPill.classList.remove('error', 'thinking');
  if (state) statusPill.classList.add(state);
  statusText.textContent = label;
}

function showEmptyState() {
  [...conversation.children].forEach((child) => {
    if (child !== emptyState) conversation.removeChild(child);
  });
  if (!conversation.contains(emptyState)) conversation.appendChild(emptyState);
  emptyState.style.display = '';
  headerTitle.textContent = 'Deep Research';
  deleteChatBtn.hidden = true;
  activeChatId = null;
  highlightActiveChat();
}

function hideEmptyState() {
  if (emptyState.parentNode === conversation) {
    emptyState.style.display = 'none';
  }
}

function clearConversation() {
  [...conversation.children].forEach((child) => {
    if (child !== emptyState) conversation.removeChild(child);
  });
}

/* ----------------- auto-hide scrollbar ----------------- */

function setupAutoHideScroll(element, idleMs = 350) {
  if (!element) return;
  let timer = null;
  element.addEventListener(
    'scroll',
    () => {
      element.classList.add('scrolling');
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        element.classList.remove('scrolling');
      }, idleMs);
    },
    { passive: true }
  );
}

setupAutoHideScroll(conversation);
setupAutoHideScroll(chatList);

/* ----------------- theme toggle ----------------- */

const themeToggle = document.getElementById('theme-toggle');

function applyTheme(theme) {
  if (theme === 'light') {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
}

let currentTheme = localStorage.getItem('theme') || 'dark';
applyTheme(currentTheme);

themeToggle.addEventListener('click', () => {
  currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
  try {
    localStorage.setItem('theme', currentTheme);
  } catch (err) {}
  applyTheme(currentTheme);
});

/* ----------------- model toggle ----------------- */

function applySelectedModel() {
  modelToggle.querySelectorAll('.model-option').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.model === selectedModel);
  });
}

modelToggle.querySelectorAll('.model-option').forEach((btn) => {
  btn.addEventListener('click', () => {
    selectedModel = btn.dataset.model;
    localStorage.setItem('selectedModel', selectedModel);
    applySelectedModel();
  });
});

applySelectedModel();

/* ----------------- sidebar ----------------- */

function isMobile() {
  return window.matchMedia('(max-width: 760px)').matches;
}

function openSidebar() {
  app.classList.remove('sidebar-hidden');
}

function closeSidebar() {
  app.classList.add('sidebar-hidden');
}

sidebarClose.addEventListener('click', closeSidebar);
sidebarToggle.addEventListener('click', openSidebar);
sidebarBackdrop.addEventListener('click', closeSidebar);

if (isMobile()) closeSidebar();

async function loadChats(query) {
  const url = query ? `/api/chats?q=${encodeURIComponent(query)}` : '/api/chats';
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to load chats');
    const data = await res.json();
    cachedChats = data.chats || [];
    renderChatList();
  } catch (err) {
    console.error(err);
  }
}

function renderChatList() {
  [...chatList.children].forEach((child) => {
    if (child !== sidebarEmpty) chatList.removeChild(child);
  });

  if (!cachedChats.length) {
    sidebarEmpty.style.display = '';
    if (searchInput.value.trim()) {
      sidebarEmpty.querySelector('p').textContent = 'No matches';
      sidebarEmpty.querySelector('span').textContent = 'Try another search';
    } else {
      sidebarEmpty.querySelector('p').textContent = 'No chats yet';
      sidebarEmpty.querySelector('span').textContent = 'Start a new research above';
    }
    return;
  }

  sidebarEmpty.style.display = 'none';
  cachedChats.forEach((chat) => {
    const item = el('div', 'chat-item');
    item.dataset.chatId = chat.id;
    if (chat.id === activeChatId) item.classList.add('active');

    item.appendChild(txt('div', 'chat-item-title', chat.title || 'Untitled'));
    const metaParts = [];
    if (chat.turn_count) metaParts.push(`${chat.turn_count} turn${chat.turn_count > 1 ? 's' : ''}`);
    if (chat.updated_at) metaParts.push(relativeTime(chat.updated_at));
    item.appendChild(txt('div', 'chat-item-meta', metaParts.join(' · ')));

    const delBtn = el(
      'button',
      'chat-item-delete',
      '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/></svg>'
    );
    delBtn.title = 'Delete chat';
    delBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      deleteChat(chat.id, chat.title);
    });
    item.appendChild(delBtn);

    item.addEventListener('click', () => openChat(chat.id));
    chatList.appendChild(item);
  });
}

function highlightActiveChat() {
  chatList.querySelectorAll('.chat-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.chatId === activeChatId);
  });
}

async function openChat(chatId) {
  if (isAsking) return;
  try {
    const res = await fetch(`/api/chats/${chatId}`);
    if (!res.ok) throw new Error('Failed to load chat');
    const chat = await res.json();
    activeChatId = chat.id;
    highlightActiveChat();
    clearConversation();
    hideEmptyState();
    headerTitle.textContent = chat.title || 'Deep Research';
    deleteChatBtn.hidden = false;

    (chat.turns || []).forEach((turn) => renderHistoricalTurn(turn));
    if (isMobile()) closeSidebar();
    scrollToBottom();
  } catch (err) {
    console.error(err);
  }
}

async function deleteChat(chatId, title) {
  const confirmed = confirm(`Delete "${title || 'this chat'}"? This cannot be undone.`);
  if (!confirmed) return;
  try {
    const res = await fetch(`/api/chats/${chatId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete chat');
    if (chatId === activeChatId) showEmptyState();
    await loadChats(searchInput.value.trim());
  } catch (err) {
    console.error(err);
  }
}

deleteChatBtn.addEventListener('click', () => {
  if (!activeChatId) return;
  const chat = cachedChats.find((c) => c.id === activeChatId);
  deleteChat(activeChatId, chat ? chat.title : '');
});

/* ----------------- search ----------------- */

searchInput.addEventListener('input', () => {
  const value = searchInput.value;
  searchClear.hidden = value.length === 0;
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(() => loadChats(value.trim()), 180);
});

searchClear.addEventListener('click', () => {
  searchInput.value = '';
  searchClear.hidden = true;
  loadChats('');
  searchInput.focus();
});

/* ----------------- new chat / suggestions ----------------- */

newChatBtn.addEventListener('click', () => {
  if (isAsking) stopResearch();
  showEmptyState();
  input.value = '';
  autosizeInput();
  if (isMobile()) closeSidebar();
  input.focus();
});

document.querySelectorAll('.suggestion').forEach((btn) => {
  btn.addEventListener('click', () => {
    input.value = btn.dataset.q || btn.textContent.trim();
    autosizeInput();
    form.requestSubmit();
  });
});

/* ----------------- composer ----------------- */

input.addEventListener('input', autosizeInput);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  if (isAsking) return;
  const question = input.value.trim();
  if (!question) return;
  startResearch(question);
});

/* ----------------- rendering helpers ----------------- */

function buildBadges({ model, fromCache, memoryTurns }) {
  const wrap = el('div', 'turn-badges');
  const modelKey = model || 'llama';
  const modelBadge = el('span', `meta-badge model-${modelKey}`);
  modelBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l3 7h7l-5.5 4 2 7L12 16l-6.5 4 2-7L2 9h7z"/></svg>`;
  modelBadge.appendChild(txt('span', null, MODEL_LABELS[modelKey] || modelKey));
  wrap.appendChild(modelBadge);

  if (fromCache) {
    const cacheBadge = el('span', 'meta-badge cache');
    cacheBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 2 13 10 19 10"/><path d="M21 12A9 9 0 1 1 12 3"/></svg>`;
    cacheBadge.appendChild(txt('span', null, 'From cache'));
    wrap.appendChild(cacheBadge);
  }

  if (memoryTurns && memoryTurns > 0) {
    const memBadge = el('span', 'meta-badge memory');
    memBadge.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>`;
    memBadge.appendChild(txt('span', null, `${memoryTurns} prior turn${memoryTurns > 1 ? 's' : ''}`));
    wrap.appendChild(memBadge);
  }

  return wrap;
}

function buildAnswerCard(markdownText) {
  const card = el('div', 'answer-card');
  if (window.marked) card.innerHTML = marked.parse(markdownText || '');
  else card.textContent = markdownText || '';
  return card;
}

function buildSection(label, count, body) {
  const section = el('div', 'section');
  const title = el('div', 'section-title');
  title.appendChild(txt('span', null, label));
  if (count != null) title.appendChild(txt('span', 'section-title-badge', String(count)));
  section.appendChild(title);
  section.appendChild(body);
  return section;
}

const TIER_META = {
  high:       { label: 'High',       cls: 'tier-high' },
  medium:     { label: 'Medium',     cls: 'tier-medium' },
  low:        { label: 'Low',        cls: 'tier-low' },
  prediction: { label: 'Prediction', cls: 'tier-prediction' },
};

function buildSourcesGrid(sources) {
  const grid = el('div', 'sources-grid');
  sources.forEach((source) => {
    const card = document.createElement('a');
    card.className = 'source-card';
    card.href = source.url;
    card.target = '_blank';
    card.rel = 'noopener noreferrer';

    const head = el('div', 'source-head');
    const fav = document.createElement('img');
    fav.className = 'source-favicon';
    fav.src = faviconUrl(source.domain || domainOf(source.url));
    fav.alt = '';
    fav.referrerPolicy = 'no-referrer';
    fav.onerror = () => {
      fav.style.visibility = 'hidden';
    };
    head.appendChild(fav);
    head.appendChild(txt('span', 'source-domain', source.domain || domainOf(source.url)));

    // Tier badge on each source card
    if (source.tier) {
      const meta = TIER_META[source.tier] || { label: source.tier, cls: 'tier-low' };
      head.appendChild(txt('span', `source-tier-badge ${meta.cls}`, meta.label));
    }
    card.appendChild(head);

    if (source.title) card.appendChild(txt('div', 'source-title', source.title));
    if (source.snippet) card.appendChild(txt('div', 'source-snippet', source.snippet));

    if (source.source_tools && source.source_tools.length) {
      const tools = el('div', 'source-tools');
      source.source_tools.forEach((tool) => tools.appendChild(txt('span', 'tool-chip', tool)));
      card.appendChild(tools);
    }

    grid.appendChild(card);
  });
  return grid;
}

function buildFactsList(facts) {
  const list = el('div', 'facts-list');
  facts.forEach((fact) => {
    const card = el('div', 'fact-card');
    card.appendChild(txt('div', 'fact-claim', fact.claim_text));

    const meta = el('div', 'fact-meta');
    const flags = fact.fact_quality_flags || {};
    if (flags.multi_source) meta.appendChild(txt('span', 'fact-flag good', 'Multi-source'));
    else if (flags.has_source) meta.appendChild(txt('span', 'fact-flag info', 'Cited'));
    if (flags.weak_evidence) meta.appendChild(txt('span', 'fact-flag warn', 'Weak evidence'));
    if (flags.needs_review) meta.appendChild(txt('span', 'fact-flag danger', 'Needs review'));

    const evidence = el('div', 'fact-evidence');
    (fact.evidence_urls || []).forEach((url) => {
      const link = document.createElement('a');
      link.className = 'evidence-link';
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = domainOf(url);
      evidence.appendChild(link);
    });
    meta.appendChild(evidence);

    card.appendChild(meta);
    list.appendChild(card);
  });
  return list;
}

function buildTrustCard(label, value, ratio) {
  const card = el('div', 'trust-card');
  card.appendChild(txt('div', 'trust-label', label));
  card.appendChild(txt('div', 'trust-value', value));
  const bar = el('div', 'trust-bar');
  const fill = el('div', 'trust-bar-fill');
  const clamped = Math.max(0, Math.min(1, ratio));
  requestAnimationFrame(() => {
    fill.style.width = clamped * 100 + '%';
  });
  bar.appendChild(fill);
  card.appendChild(bar);
  return card;
}

function buildTrustGrid(signals) {
  const wrap = el('div', 'trust-wrap');

  const grid = el('div', 'trust-grid');

  // Source quality: weighted average tier of all retrieved sources (high=1.0 … prediction=0.02)
  const sqScore = signals.source_quality_score || 0;
  grid.appendChild(buildTrustCard('Source quality', Math.round(sqScore * 100) + '%', sqScore));

  // Citation strength: how authoritative are the sources backing each claim
  const strength = signals.citation_strength || 0;
  grid.appendChild(buildTrustCard('Citation strength', Math.round(strength * 100) + '%', strength));

  // What fraction of claims have at least one source
  grid.appendChild(
    buildTrustCard(
      'Citation coverage',
      Math.round((signals.citation_coverage || 0) * 100) + '%',
      signals.citation_coverage || 0
    )
  );

  // What fraction of claims are backed by ≥2 independent domains
  grid.appendChild(
    buildTrustCard(
      'Multi-source claims',
      Math.round((signals.multi_source_claim_ratio || 0) * 100) + '%',
      signals.multi_source_claim_ratio || 0
    )
  );

  wrap.appendChild(grid);

  // Tier breakdown row
  const breakdown = signals.tier_breakdown;
  if (breakdown) {
    const row = el('div', 'tier-breakdown');
    [
      { key: 'high',       label: 'High',       cls: 'tier-high' },
      { key: 'medium',     label: 'Medium',     cls: 'tier-medium' },
      { key: 'low',        label: 'Low',        cls: 'tier-low' },
      { key: 'prediction', label: 'Prediction', cls: 'tier-prediction' },
    ].forEach(({ key, label, cls }) => {
      const count = (breakdown[key] || 0);
      if (count === 0) return;   // hide zero-count tiers to save space
      const chip = el('span', `tier-chip ${cls}`);
      chip.innerHTML = `<span class="tier-chip-dot"></span>${count} ${label}`;
      row.appendChild(chip);
    });
    if (signals.total_sources) {
      row.appendChild(txt('span', 'tier-total', `${signals.total_sources} source${signals.total_sources !== 1 ? 's' : ''} retrieved`));
    }
    wrap.appendChild(row);
  }

  return wrap;
}

function renderHistoricalTurn(turn) {
  const turnEl = el('div', 'turn');
  turnEl.appendChild(txt('div', 'user-msg', turn.question || ''));

  const agentBlock = el('div', 'agent-block');
  agentBlock.appendChild(
    buildBadges({ model: turn.model, fromCache: !!turn.from_cache, memoryTurns: 0 })
  );
  agentBlock.appendChild(buildAnswerCard(turn.answer || ''));

  if (turn.sources && turn.sources.length) {
    agentBlock.appendChild(buildSection('Sources', turn.sources.length, buildSourcesGrid(turn.sources)));
  }
  if (turn.facts && turn.facts.length) {
    agentBlock.appendChild(buildSection('Extracted facts', turn.facts.length, buildFactsList(turn.facts)));
  }
  if (turn.trust_signals) {
    agentBlock.appendChild(buildSection('Trust signals', null, buildTrustGrid(turn.trust_signals)));
  }

  turnEl.appendChild(agentBlock);
  conversation.appendChild(turnEl);
}

/* ----------------- streaming ----------------- */

function startResearch(question) {
  isAsking = true;
  setSubmitMode('stop');
  input.value = '';
  autosizeInput();
  setStatus('thinking', 'Researching');
  hideEmptyState();

  const turn = el('div', 'turn');
  turn.appendChild(txt('div', 'user-msg', question));

  const agentBlock = el('div', 'agent-block');
  let badgesEl = null;
  turn.appendChild(agentBlock);

  const trace = el('div', 'trace');
  const traceHeader = el('div', 'trace-header');
  const traceTitle = el('div', 'trace-title');
  const spinner = el('span', 'trace-spinner');
  activeTraceSpinner = spinner;
  traceTitle.appendChild(spinner);
  const traceLabel = txt('span', null, 'Planning research…');
  activeTraceLabel = traceLabel;
  traceTitle.appendChild(traceLabel);
  traceHeader.appendChild(traceTitle);
  const chevron = el(
    'span',
    'trace-chevron',
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>'
  );
  traceHeader.appendChild(chevron);
  trace.appendChild(traceHeader);
  const traceBody = el('div', 'trace-body');
  trace.appendChild(traceBody);
  traceHeader.addEventListener('click', () => trace.classList.toggle('collapsed'));
  agentBlock.appendChild(trace);

  conversation.appendChild(turn);
  scrollToBottom();

  const params = new URLSearchParams({ question, model: selectedModel });
  if (activeChatId) params.set('chat_id', activeChatId);
  const url = `/ask_agent_stream?${params.toString()}`;

  const es = new EventSource(url);
  activeSource = es;
  const startTime = Date.now();
  const pendingTools = new Map();
  let streamedFromCache = false;
  let streamedMemoryTurns = 0;
  let streamedModel = selectedModel;

  function ensureBadges() {
    if (badgesEl) {
      const fresh = buildBadges({
        model: streamedModel,
        fromCache: streamedFromCache,
        memoryTurns: streamedMemoryTurns,
      });
      badgesEl.replaceWith(fresh);
      badgesEl = fresh;
    } else {
      badgesEl = buildBadges({
        model: streamedModel,
        fromCache: streamedFromCache,
        memoryTurns: streamedMemoryTurns,
      });
      agentBlock.insertBefore(badgesEl, trace);
    }
  }

  es.addEventListener('start', (event) => {
    const data = JSON.parse(event.data);
    if (data.chat_id) {
      activeChatId = data.chat_id;
      headerTitle.textContent = data.question || 'Deep Research';
      deleteChatBtn.hidden = false;
    }
    streamedModel = data.model || selectedModel;
    streamedMemoryTurns = data.memory_turns || 0;
    ensureBadges();
    traceLabel.textContent = streamedMemoryTurns > 0
      ? 'Planning with conversation context…'
      : 'Planning research…';
  });

  es.addEventListener('cached', (event) => {
    const data = JSON.parse(event.data);
    streamedFromCache = true;
    ensureBadges();
    if (activeTraceSpinner) { activeTraceSpinner.remove(); activeTraceSpinner = null; }
    const icon = el(
      'span',
      'trace-cache-icon',
      '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 2 13 10 19 10"/><path d="M21 12A9 9 0 1 1 12 3"/></svg>'
    );
    traceTitle.insertBefore(icon, traceLabel);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
    traceLabel.textContent = `Served from cache · ${elapsed}s`;
    trace.classList.add('collapsed');
  });

  es.addEventListener('tool_call', (event) => {
    const data = JSON.parse(event.data);
    const step = el('div', 'trace-step');
    const head = el('div', 'trace-step-head');
    head.appendChild(txt('span', 'trace-tool', data.name));
    step.appendChild(head);

    let argText = '';
    if (data.args && typeof data.args === 'object') {
      const value = data.args.query || data.args.url || data.args.input || JSON.stringify(data.args);
      argText = typeof value === 'string' ? value : JSON.stringify(value);
    } else if (data.args) {
      argText = String(data.args);
    }
    if (argText) step.appendChild(txt('div', 'trace-step-detail', argText));

    traceBody.appendChild(step);
    if (data.id) pendingTools.set(data.id, step);
    pendingTools.set(data.name + ':last', step);
    traceLabel.textContent = `Running ${data.name}…`;
    scrollToBottom();
  });

  es.addEventListener('tool_result', (event) => {
    const data = JSON.parse(event.data);
    let step = data.id ? pendingTools.get(data.id) : null;
    if (!step) step = pendingTools.get(data.name + ':last');
    const result = txt('div', 'trace-step-result', `${data.length.toLocaleString()} characters returned`);
    if (step) step.appendChild(result);
    else traceBody.appendChild(result);
    scrollToBottom();
  });

  es.addEventListener('answer', (event) => {
    const data = JSON.parse(event.data);
    traceLabel.textContent = 'Drafting answer…';
    let answerCard = agentBlock.querySelector('.answer-card');
    if (!answerCard) {
      answerCard = buildAnswerCard(data.text);
      agentBlock.appendChild(answerCard);
    } else if (window.marked) {
      answerCard.innerHTML = marked.parse(data.text || '');
    } else {
      answerCard.textContent = data.text || '';
    }
    scrollToBottom();
  });

  es.addEventListener('verifying', () => {
    traceLabel.textContent = 'Fact-checking answer against sources…';
  });

  es.addEventListener('extracting', () => {
    traceLabel.textContent = 'Extracting facts & verifying citations…';
  });

  es.addEventListener('final', (event) => {
    const data = JSON.parse(event.data);
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(streamedFromCache ? 2 : 1);

    if (!streamedFromCache) {
      if (activeTraceSpinner) { activeTraceSpinner.remove(); activeTraceSpinner = null; }
      const check = el(
        'span',
        'trace-check',
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
      );
      traceTitle.insertBefore(check, traceLabel);
      traceLabel.textContent = `Research complete · ${elapsed}s`;
      trace.classList.add('collapsed');
    }

    streamedFromCache = !!data.from_cache;
    streamedModel = data.model || streamedModel;
    ensureBadges();

    let answerCard = agentBlock.querySelector('.answer-card');
    if (!answerCard) {
      answerCard = buildAnswerCard(data.answer);
      agentBlock.appendChild(answerCard);
    } else if (window.marked) {
      answerCard.innerHTML = marked.parse(data.answer || '');
    } else {
      answerCard.textContent = data.answer || '';
    }

    if (data.sources && data.sources.length) {
      agentBlock.appendChild(buildSection('Sources', data.sources.length, buildSourcesGrid(data.sources)));
    }
    if (data.facts && data.facts.length) {
      agentBlock.appendChild(buildSection('Extracted facts', data.facts.length, buildFactsList(data.facts)));
    }
    if (data.trust_signals) {
      agentBlock.appendChild(buildSection('Trust signals', null, buildTrustGrid(data.trust_signals)));
    }

    if (data.chat_title) headerTitle.textContent = data.chat_title;
    scrollToBottom();
  });

  es.addEventListener('done', () => {
    es.close();
    activeSource = null;
    isAsking = false;
    setSubmitMode('send');
    setStatus(null, 'Online');
    loadChats(searchInput.value.trim());
    input.focus();
  });

  es.addEventListener('error', (event) => {
    let message = 'Connection error or agent failure.';
    try {
      if (event.data) message = JSON.parse(event.data).message || message;
    } catch {}
    if (activeTraceSpinner) { activeTraceSpinner.remove(); activeTraceSpinner = null; }
    if (activeTraceLabel) { activeTraceLabel.textContent = 'Failed'; activeTraceLabel = null; }
    const banner = txt('div', 'error-banner', message);
    agentBlock.appendChild(banner);
    es.close();
    activeSource = null;
    isAsking = false;
    setSubmitMode('send');
    setStatus('error', 'Error');
    scrollToBottom();
  });
}

/* ----------------- boot ----------------- */

autosizeInput();
input.focus();
loadChats();
