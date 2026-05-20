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
const settingsBtn = document.getElementById('settings-btn');

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

// Single click handler — routes to stop or submit based on isAsking.
// Button is type="button" so it NEVER auto-submits the form. We own the flow.
function handleSubmitClick(e) {
  if (e) e.preventDefault();
  if (isAsking) {
    stopResearch();
    return;
  }
  const question = input.value.trim();
  if (!question) return;
  startResearch(question);
}

submitBtn.addEventListener('click', handleSubmitClick);

// Escape key shortcut — close any open modal first, otherwise stop the search
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (settingsModal && settingsModal.classList.contains('open')) {
    e.preventDefault();
    closeSettings();
    return;
  }
  if (answerBreakdownModal && answerBreakdownModal.classList.contains('open')) {
    e.preventDefault();
    closeAnswerBreakdown();
    return;
  }
  if (trustExplainerModal && trustExplainerModal.classList.contains('open')) {
    e.preventDefault();
    closeTrustExplainer();
    return;
  }
  if (isAsking) {
    e.preventDefault();
    stopResearch();
  }
});

if (settingsBtn) {
  settingsBtn.addEventListener('click', openSettings);
}

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
    // Attach a regenerate button to the most recent rendered turn.
    const turnEls = conversation.querySelectorAll('.turn');
    if (turnEls.length && chat.turns && chat.turns.length) {
      const lastTurnEl = turnEls[turnEls.length - 1];
      const lastQ = chat.turns[chat.turns.length - 1].question;
      const agentBlock = lastTurnEl.querySelector('.agent-block');
      if (lastQ && agentBlock) {
        agentBlock.appendChild(buildRegenerateButton(lastQ));
      }
    }
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

/* ----------------- regenerate ----------------- */

function buildRegenerateButton(question) {
  const btn = el('button', 'regenerate-btn');
  btn.type = 'button';
  btn.title = 'Regenerate this answer with the currently-selected model';
  btn.setAttribute('aria-label', 'Regenerate answer');
  btn.innerHTML = `
    <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="23 4 23 10 17 10"/>
      <polyline points="1 20 1 14 7 14"/>
      <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
    </svg>
    <span>Regenerate</span>
  `;
  btn.dataset.question = question || '';
  btn.addEventListener('click', () => {
    if (btn.disabled) return;
    regenerateLastTurn(btn.dataset.question);
  });
  return btn;
}

function removeAllRegenerateButtons() {
  conversation.querySelectorAll('.regenerate-btn').forEach((b) => b.remove());
}

function regenerateLastTurn(question) {
  if (isAsking) return;
  if (!question || !question.trim()) return;
  // Remove the current last turn from the DOM. The backend still has it
  // until the new answer completes — chat_store.replace_last_turn does the
  // swap atomically only on success, so a cancelled regenerate leaves the
  // stored data intact (visible again on reload).
  const turns = [...conversation.querySelectorAll('.turn')];
  if (turns.length) turns[turns.length - 1].remove();
  removeAllRegenerateButtons();
  startResearch(question, { regenerate: true });
}

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

function buildSection(label, count, body, actionEl) {
  const section = el('div', 'section');
  const title = el('div', 'section-title');
  title.appendChild(txt('span', null, label));
  if (count != null) title.appendChild(txt('span', 'section-title-badge', String(count)));
  if (actionEl) title.appendChild(actionEl);
  section.appendChild(title);
  section.appendChild(body);
  return section;
}

/* ----------------- trust signals explainer ----------------- */

let trustExplainerModal = null;

function buildTrustHelpButton() {
  const btn = el('button', 'section-help-btn');
  btn.type = 'button';
  btn.title = 'How are trust signals calculated?';
  btn.setAttribute('aria-label', 'Explain trust signals');
  btn.innerHTML = `
    <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="9"/>
      <path d="M12 8h.01"/>
      <path d="M11 12h1v5h1"/>
    </svg>
    <span>How is this scored?</span>
  `;
  btn.addEventListener('click', openTrustExplainer);
  return btn;
}

function openTrustExplainer() {
  if (!trustExplainerModal) {
    trustExplainerModal = buildTrustExplainerModal();
    document.body.appendChild(trustExplainerModal);
  }
  trustExplainerModal.classList.add('open');
  document.body.classList.add('modal-open');
}

function closeTrustExplainer() {
  if (trustExplainerModal) {
    trustExplainerModal.classList.remove('open');
  }
  document.body.classList.remove('modal-open');
}

function buildTrustExplainerModal() {
  const backdrop = el('div', 'modal-backdrop trust-explainer-backdrop');
  backdrop.addEventListener('click', closeTrustExplainer);

  const modal = el('div', 'modal');
  modal.addEventListener('click', (e) => e.stopPropagation());

  modal.innerHTML = `
    <button class="modal-close" type="button" aria-label="Close">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 6L6 18M6 6l12 12"/>
      </svg>
    </button>
    <div class="modal-header">
      <h2>How trust signals are calculated</h2>
      <p>Every answer is scored on four dimensions based on the quality and consistency of the sources the agent retrieved.</p>
    </div>
    <div class="modal-body">
      <div class="explainer-h3">Source reputation tiers</div>
      <p class="explainer-intro">
        Every URL the agent retrieves is classified into one of four reputation tiers.
        The tier determines the source's weight in the trust score.
      </p>
      <div class="explainer-tier-grid">
        <div class="explainer-tier">
          <div class="explainer-tier-head">
            <span class="tier-chip-dot" style="background: var(--success)"></span>
            <span class="explainer-tier-name tier-high-text">HIGH</span>
            <span class="explainer-tier-weight">weight 1.00</span>
          </div>
          <p>Wikipedia, official org sites (.gov, .edu, who.int, fifa.com, uefa.com), major newswires (Reuters, AP, AFP), major newspapers (BBC, NYT, Guardian, FT), peer-reviewed journals (Nature, Lancet).</p>
        </div>
        <div class="explainer-tier">
          <div class="explainer-tier-head">
            <span class="tier-chip-dot" style="background: var(--accent)"></span>
            <span class="explainer-tier-name tier-medium-text">MEDIUM</span>
            <span class="explainer-tier-weight">weight 0.65</span>
          </div>
          <p>Mainstream news (CNN, NBC, Bloomberg, Axios), established sports / tech publications (ESPN, TechCrunch, Wired, The Verge), business outlets (Forbes, CNBC).</p>
        </div>
        <div class="explainer-tier">
          <div class="explainer-tier-head">
            <span class="tier-chip-dot" style="background: var(--warning)"></span>
            <span class="explainer-tier-name tier-low-text">LOW</span>
            <span class="explainer-tier-weight">weight 0.35</span>
          </div>
          <p>Unknown blogs, small or unfamiliar sites, content farms. Used only when no higher-tier source is available, with the agent instructed to corroborate before relying on them.</p>
        </div>
        <div class="explainer-tier">
          <div class="explainer-tier-head">
            <span class="tier-chip-dot" style="background: var(--danger)"></span>
            <span class="explainer-tier-name tier-prediction-text">PREDICTION</span>
            <span class="explainer-tier-weight">weight 0.02</span>
          </div>
          <p>Betting / odds sites (bet365, betfair, oddschecker), "who will win" articles, upcoming-event previews. These describe events that haven't happened yet and are explicitly excluded as factual evidence.</p>
        </div>
      </div>

      <div class="explainer-h3">How tiers are assigned</div>
      <ol class="explainer-list">
        <li>Curated allowlists of trusted high-tier and medium-tier domains</li>
        <li>Top-level domain heuristics — <code>.gov</code>, <code>.edu</code>, <code>.mil</code>, <code>.ac.uk</code> → HIGH</li>
        <li>Regex patterns on title and snippet that flag prediction content — "who will win", "odds", "predictions", "preview", "ahead of", "set to win"</li>
        <li>Substring matches on betting-related domain names — "odds", "betting", "betfair", "draftkings", "tipster"</li>
        <li>Anything else defaults to LOW</li>
      </ol>

      <div class="explainer-h3">The four metrics</div>

      <div class="explainer-metric">
        <div class="explainer-metric-head">
          <span class="explainer-metric-name">Source quality</span>
          <code class="explainer-metric-formula">mean(tier_weight(s) for s in sources)</code>
        </div>
        <p>The average reputation tier across <strong>all</strong> sources retrieved during research, scaled to 0–100%. If most results came from Wikipedia and Reuters, this is high; if they came from random blogs or betting sites, it drops.</p>
        <p class="why">Why it matters — Tells you whether the answer drew from authoritative references or from low-quality material.</p>
      </div>

      <div class="explainer-metric">
        <div class="explainer-metric-head">
          <span class="explainer-metric-name">Citation strength</span>
          <code class="explainer-metric-formula">mean( max(tier_weight(s) for s in claim.evidence) for claim in claims )</code>
        </div>
        <p>For each individual claim, take the highest-tier source backing it. Then average across all claims. Rewards <strong>quality over quantity</strong>.</p>
        <p class="why">Why it matters — A claim backed by one Wikipedia source is more trustworthy than one backed by five unknown blogs. This metric captures that.</p>
      </div>

      <div class="explainer-metric">
        <div class="explainer-metric-head">
          <span class="explainer-metric-name">Citation coverage</span>
          <code class="explainer-metric-formula">claims_with_sources / total_claims</code>
        </div>
        <p>The fraction of factual claims in the answer that have at least one supporting source attached.</p>
        <p class="why">Why it matters — Detects unsupported assertions. 100% means every claim is backed by at least one citation; lower numbers reveal claims pulled from the model's own memory.</p>
      </div>

      <div class="explainer-metric">
        <div class="explainer-metric-head">
          <span class="explainer-metric-name">Multi-source claims</span>
          <code class="explainer-metric-formula">claims_with_≥2_distinct_domains / total_claims</code>
        </div>
        <p>The fraction of claims corroborated by at least two <strong>independent domains</strong>. Two URLs from the same site don't count — different publishers required.</p>
        <p class="why">Why it matters — A claim confirmed by multiple independent outlets is more robust than one appearing on a single site.</p>
      </div>

      <div class="explainer-h3">Reading the numbers</div>
      <p class="explainer-intro">
        Source quality and citation strength are tier-weight averages scaled to 0–100% (100% = every source is HIGH tier).
        Citation coverage and multi-source ratios are plain fractions of the total claim count.
        Below the four cards, the tier breakdown chips show the exact mix of HIGH/MEDIUM/LOW/PREDICTION sources used.
      </p>
    </div>
  `;

  modal.querySelector('.modal-close').addEventListener('click', closeTrustExplainer);

  backdrop.appendChild(modal);
  return backdrop;
}

/* ----------------- per-answer trust breakdown ----------------- */

const TIER_WEIGHTS = { high: 1.0, medium: 0.65, low: 0.35, prediction: 0.02 };
const TIER_ORDER   = { high: 0, medium: 1, low: 2, prediction: 3 };

function tierWeight(tier) {
  const w = TIER_WEIGHTS[tier];
  return typeof w === 'number' ? w : 0.35;
}

function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function truncate(s, max) {
  const str = String(s || '');
  if (str.length <= max) return str;
  return str.slice(0, max).trim() + '…';
}

let answerBreakdownModal = null;

function buildAnswerBreakdownButton(signals, sources, facts) {
  const btn = el('button', 'section-help-btn breakdown-btn');
  btn.type = 'button';
  btn.title = "Show how this answer's trust signals were computed";
  btn.setAttribute('aria-label', 'Show breakdown for this answer');
  btn.innerHTML = `
    <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="14" width="4" height="7"/>
      <rect x="10" y="9" width="4" height="12"/>
      <rect x="17" y="4" width="4" height="17"/>
    </svg>
    <span>Show the math</span>
  `;
  btn.addEventListener('click', () => openAnswerBreakdown(signals, sources, facts));
  return btn;
}

function buildTrustActions(signals, sources, facts) {
  const wrap = el('div', 'section-actions');
  wrap.appendChild(buildTrustHelpButton());
  if (signals && Array.isArray(sources) && Array.isArray(facts)) {
    wrap.appendChild(buildAnswerBreakdownButton(signals, sources, facts));
  }
  return wrap;
}

function openAnswerBreakdown(signals, sources, facts) {
  closeAnswerBreakdown();
  answerBreakdownModal = buildAnswerBreakdownModal(signals, sources, facts);
  document.body.appendChild(answerBreakdownModal);
  // Force reflow then add open class for animation
  void answerBreakdownModal.offsetWidth;
  answerBreakdownModal.classList.add('open');
  document.body.classList.add('modal-open');
}

function closeAnswerBreakdown() {
  if (answerBreakdownModal) {
    answerBreakdownModal.remove();
    answerBreakdownModal = null;
  }
  if (!trustExplainerModal || !trustExplainerModal.classList.contains('open')) {
    document.body.classList.remove('modal-open');
  }
}

function buildAnswerBreakdownModal(signals, sources, facts) {
  const safeSignals = signals || {};
  const safeSources = Array.isArray(sources) ? sources : [];
  const safeFacts   = Array.isArray(facts)   ? facts   : [];

  // url → source lookup for fast tier resolution
  const sourceByUrl = new Map();
  safeSources.forEach((s) => { if (s && s.url) sourceByUrl.set(s.url, s); });

  const backdrop = el('div', 'modal-backdrop breakdown-backdrop');
  backdrop.addEventListener('click', closeAnswerBreakdown);

  const modal = el('div', 'modal breakdown-modal');
  modal.addEventListener('click', (e) => e.stopPropagation());

  modal.innerHTML = `
    <button class="modal-close" type="button" aria-label="Close">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 6L6 18M6 6l12 12"/>
      </svg>
    </button>
    <div class="modal-header">
      <h2>Breakdown of this answer</h2>
      <p>The actual values that went into the trust signals shown above — every source, every claim, and every step of the arithmetic.</p>
    </div>
    <div class="modal-body">
      ${renderSourceQualityBreakdown(safeSignals, safeSources)}
      ${renderCitationStrengthBreakdown(safeSignals, safeFacts, sourceByUrl)}
      ${renderCitationCoverageBreakdown(safeSignals, safeFacts)}
      ${renderMultiSourceBreakdown(safeSignals, safeFacts)}
    </div>
  `;

  modal.querySelector('.modal-close').addEventListener('click', closeAnswerBreakdown);

  backdrop.appendChild(modal);
  return backdrop;
}

function renderSourceQualityBreakdown(signals, sources) {
  const score   = signals.source_quality_score || 0;
  const scorePct = Math.round(score * 100);

  if (sources.length === 0) {
    return breakdownEmpty('Source quality', scorePct, 'mean( tier_weight(s) for s in sources )', 'No sources were retrieved for this answer.');
  }

  // Sort: high → medium → low → prediction
  const sorted = [...sources].sort((a, b) => (TIER_ORDER[a.tier] ?? 99) - (TIER_ORDER[b.tier] ?? 99));
  const sum = sorted.reduce((acc, s) => acc + tierWeight(s.tier), 0);
  const avg = sum / sorted.length;
  const weightsStr = sorted.map((s) => tierWeight(s.tier).toFixed(2)).join(' + ');
  const formula = `(${weightsStr}) ÷ ${sorted.length} = ${avg.toFixed(3)}`;

  const rows = sorted.map((s) => {
    const tier      = s.tier || 'low';
    const tierLabel = tier.toUpperCase();
    const weight    = tierWeight(tier).toFixed(2);
    const domain    = s.domain || domainOf(s.url || '');
    return `
      <div class="breakdown-row">
        <span class="breakdown-row-tag tier-${tier}">${tierLabel}</span>
        <span class="breakdown-row-weight">${weight}</span>
        <span class="breakdown-row-domain" title="${escapeHtml(domain)}">${escapeHtml(domain)}</span>
        <span class="breakdown-row-title" title="${escapeHtml(s.title || '')}">${escapeHtml(s.title || '—')}</span>
      </div>
    `;
  }).join('');

  return `
    <div class="breakdown-section">
      ${breakdownHead('Source quality', scorePct)}
      ${breakdownFormula(formula)}
      <p class="breakdown-explainer">Weighted average across ${sorted.length} retrieved source${sorted.length === 1 ? '' : 's'}, sorted by tier.</p>
      <div class="breakdown-rows">${rows}</div>
    </div>
  `;
}

function renderCitationStrengthBreakdown(signals, facts, sourceByUrl) {
  const score = signals.citation_strength || 0;
  const scorePct = Math.round(score * 100);

  if (facts.length === 0) {
    return breakdownEmpty('Citation strength', scorePct, 'mean( max(tier_weight(s)) for s in claim.evidence )', 'No claims were extracted from this answer.');
  }

  const perClaim = facts.map((fact) => {
    let bestWeight = 0;
    let bestTier = null;
    let bestSource = null;
    (fact.evidence_urls || []).forEach((url) => {
      const s = sourceByUrl.get(url);
      if (s) {
        const w = tierWeight(s.tier);
        if (w > bestWeight) {
          bestWeight = w;
          bestTier = s.tier;
          bestSource = s;
        }
      }
    });
    return { fact, bestWeight, bestTier, bestSource };
  });

  const sum = perClaim.reduce((acc, c) => acc + c.bestWeight, 0);
  const avg = sum / perClaim.length;
  const weightsStr = perClaim.map((c) => c.bestWeight.toFixed(2)).join(' + ');
  const formula = `(${weightsStr}) ÷ ${perClaim.length} = ${avg.toFixed(3)}`;

  const rows = perClaim.map((c, idx) => {
    const claim = truncate(c.fact.claim_text, 140);
    const cls = c.bestSource ? 'cited' : 'uncited';
    if (c.bestSource) {
      const tier = c.bestTier || 'low';
      const domain = c.bestSource.domain || domainOf(c.bestSource.url || '');
      return `
        <div class="breakdown-claim-row ${cls}">
          <span class="breakdown-claim-num">#${idx + 1}</span>
          <span class="breakdown-claim-text">${escapeHtml(claim)}</span>
          <span class="breakdown-claim-best">
            <span class="breakdown-row-tag tier-${tier}">${tier.toUpperCase()}</span>
            <span class="breakdown-claim-domain">${escapeHtml(domain)}</span>
          </span>
          <span class="breakdown-claim-weight">${c.bestWeight.toFixed(2)}</span>
        </div>
      `;
    }
    return `
      <div class="breakdown-claim-row ${cls}">
        <span class="breakdown-claim-num">#${idx + 1}</span>
        <span class="breakdown-claim-text">${escapeHtml(claim)}</span>
        <span class="breakdown-claim-best uncited">no supporting source</span>
        <span class="breakdown-claim-weight">0.00</span>
      </div>
    `;
  }).join('');

  return `
    <div class="breakdown-section">
      ${breakdownHead('Citation strength', scorePct)}
      ${breakdownFormula(formula)}
      <p class="breakdown-explainer">For each of the ${perClaim.length} claim${perClaim.length === 1 ? '' : 's'}, the highest-tier supporting source contributes its weight. Then averaged.</p>
      <div class="breakdown-claims">${rows}</div>
    </div>
  `;
}

function renderCitationCoverageBreakdown(signals, facts) {
  const score = signals.citation_coverage || 0;
  const scorePct = Math.round(score * 100);

  if (facts.length === 0) {
    return breakdownEmpty('Citation coverage', scorePct, 'claims_with_sources ÷ total_claims', 'No claims were extracted from this answer.');
  }

  const cited = facts.filter((f) => (f.evidence_urls || []).length > 0).length;
  const total = facts.length;
  const formula = `${cited} ÷ ${total} = ${(cited / total).toFixed(3)}`;

  const rows = facts.map((fact, idx) => {
    const urls = fact.evidence_urls || [];
    const count = urls.length;
    const cls = count > 0 ? 'cited' : 'uncited';
    const status = count > 0 ? '✓' : '✗';
    const label = count > 0 ? `${count} source${count === 1 ? '' : 's'}` : 'no sources';
    return `
      <div class="breakdown-claim-row ${cls}">
        <span class="breakdown-claim-status">${status}</span>
        <span class="breakdown-claim-num">#${idx + 1}</span>
        <span class="breakdown-claim-text">${escapeHtml(truncate(fact.claim_text, 140))}</span>
        <span class="breakdown-claim-count ${cls}">${label}</span>
      </div>
    `;
  }).join('');

  return `
    <div class="breakdown-section">
      ${breakdownHead('Citation coverage', scorePct)}
      ${breakdownFormula(formula)}
      <p class="breakdown-explainer">${cited} of ${total} claim${total === 1 ? '' : 's'} have at least one supporting source attached.</p>
      <div class="breakdown-claims">${rows}</div>
    </div>
  `;
}

function renderMultiSourceBreakdown(signals, facts) {
  const score = signals.multi_source_claim_ratio || 0;
  const scorePct = Math.round(score * 100);

  if (facts.length === 0) {
    return breakdownEmpty('Multi-source claims', scorePct, 'claims_with_≥2_distinct_domains ÷ total_claims', 'No claims were extracted from this answer.');
  }

  const perClaim = facts.map((fact) => {
    const urls = fact.evidence_urls || [];
    const domains = new Set();
    urls.forEach((url) => {
      try {
        const d = new URL(url).hostname.replace(/^www\./, '');
        if (d) domains.add(d);
      } catch (_) { /* ignore bad URLs */ }
    });
    return { fact, domains: [...domains] };
  });

  const multi = perClaim.filter((c) => c.domains.length >= 2).length;
  const total = perClaim.length;
  const formula = `${multi} ÷ ${total} = ${(multi / total).toFixed(3)}`;

  const rows = perClaim.map((c, idx) => {
    const count = c.domains.length;
    const cls = count >= 2 ? 'cited' : 'uncited';
    const status = count >= 2 ? '✓' : '✗';
    const chips = count > 0
      ? c.domains.map((d) => `<span class="breakdown-mini-domain">${escapeHtml(d)}</span>`).join('')
      : '<span class="breakdown-claim-count uncited">no domains</span>';
    return `
      <div class="breakdown-claim-row ${cls}">
        <span class="breakdown-claim-status">${status}</span>
        <span class="breakdown-claim-num">#${idx + 1}</span>
        <span class="breakdown-claim-text">${escapeHtml(truncate(c.fact.claim_text, 140))}</span>
        <span class="breakdown-claim-domains">${chips}</span>
      </div>
    `;
  }).join('');

  return `
    <div class="breakdown-section">
      ${breakdownHead('Multi-source claims', scorePct)}
      ${breakdownFormula(formula)}
      <p class="breakdown-explainer">${multi} of ${total} claim${total === 1 ? '' : 's'} corroborated by ≥2 independent domains.</p>
      <div class="breakdown-claims">${rows}</div>
    </div>
  `;
}

function breakdownHead(name, scorePct) {
  return `
    <div class="breakdown-section-head">
      <span class="breakdown-section-name">${name}</span>
      <span class="breakdown-section-value">${scorePct}%</span>
    </div>
  `;
}

function breakdownFormula(text) {
  return `
    <p class="breakdown-formula">
      <span class="breakdown-label">Math:</span>
      <code>${text}</code>
    </p>
  `;
}

function breakdownEmpty(name, scorePct, formula, explainer) {
  return `
    <div class="breakdown-section">
      ${breakdownHead(name, scorePct)}
      ${breakdownFormula(formula)}
      <p class="breakdown-empty">${explainer}</p>
    </div>
  `;
}

/* ----------------- settings ----------------- */

let settingsModal = null;

async function fetchSettings() {
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) throw new Error('failed');
    return await res.json();
  } catch (err) {
    console.error('Failed to load settings:', err);
    return null;
  }
}

async function patchSettings(patch) {
  try {
    const res = await fetch('/api/settings', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!res.ok) throw new Error('failed');
    return await res.json();
  } catch (err) {
    console.error('Failed to save settings:', err);
    return null;
  }
}

async function resetSettings() {
  try {
    const res = await fetch('/api/settings/reset', { method: 'POST' });
    if (!res.ok) throw new Error('failed');
    return await res.json();
  } catch (err) {
    console.error('Failed to reset settings:', err);
    return null;
  }
}

async function clearAnswerCache() {
  try {
    const res = await fetch('/api/cache/clear', { method: 'POST' });
    if (!res.ok) throw new Error('failed');
    return await res.json();
  } catch (err) {
    console.error('Failed to clear cache:', err);
    return null;
  }
}

async function openSettings() {
  const data = await fetchSettings();
  if (!data) return;
  closeSettings();
  settingsModal = buildSettingsModal(data.settings, data.defaults);
  document.body.appendChild(settingsModal);
  void settingsModal.offsetWidth;
  settingsModal.classList.add('open');
  document.body.classList.add('modal-open');
}

function closeSettings() {
  if (settingsModal) {
    settingsModal.remove();
    settingsModal = null;
  }
  // Only release body scroll lock if no other modals are open
  const otherOpen =
    (trustExplainerModal && trustExplainerModal.classList.contains('open')) ||
    (answerBreakdownModal && answerBreakdownModal.classList.contains('open'));
  if (!otherOpen) document.body.classList.remove('modal-open');
}

function showSettingsToast(modal, message, kind) {
  const toast = modal.querySelector('#setting-toast');
  if (!toast) return;
  toast.textContent = message;
  toast.className = 'setting-toast ' + (kind || 'success');
  toast.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.hidden = true; }, 2400);
}

function buildSettingsModal(settings, defaults) {
  const backdrop = el('div', 'modal-backdrop settings-backdrop');
  backdrop.addEventListener('click', closeSettings);

  const modal = el('div', 'modal settings-modal');
  modal.addEventListener('click', (e) => e.stopPropagation());

  const cacheHours = Math.round((settings.cache_ttl_seconds || 0) / 3600);
  const historyLimit = settings.history_turn_limit ?? 5;
  const cacheDefaultH = Math.round((defaults.cache_ttl_seconds || 0) / 3600);

  modal.innerHTML = `
    <button class="modal-close" type="button" aria-label="Close">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
    </button>
    <div class="modal-header">
      <h2>Settings</h2>
      <p>Tune the agent's behavior and manage local data. Changes save automatically.</p>
    </div>
    <div class="modal-body">
      <div class="explainer-h3">Performance &amp; behavior</div>

      <div class="setting-row">
        <div class="setting-label">
          <span class="setting-name">Cache TTL</span>
          <span class="setting-desc">How long cached answers stay valid. Set to <code>0</code> to disable caching entirely. <span class="setting-default">Default: ${cacheDefaultH}h</span></span>
        </div>
        <div class="setting-control">
          <input type="number" id="setting-cache-ttl" value="${cacheHours}" min="0" max="720" class="setting-input"/>
          <span class="setting-unit">hours</span>
        </div>
      </div>

      <div class="setting-row">
        <div class="setting-label">
          <span class="setting-name">History limit</span>
          <span class="setting-desc">How many prior turns are passed back to the agent as conversation context. Set to <code>0</code> to disable memory. <span class="setting-default">Default: ${defaults.history_turn_limit}</span></span>
        </div>
        <div class="setting-control">
          <input type="number" id="setting-history" value="${historyLimit}" min="0" max="20" class="setting-input"/>
          <span class="setting-unit">turns</span>
        </div>
      </div>

      <div class="setting-row">
        <div class="setting-label">
          <span class="setting-name">Verification pass</span>
          <span class="setting-desc">A second LLM call re-checks each draft answer against the search results and rewrites unsupported claims. Adds ~5–10s but catches hallucinations.</span>
        </div>
        <div class="setting-control">
          <label class="toggle">
            <input type="checkbox" id="setting-verification" ${settings.verification_enabled ? 'checked' : ''}/>
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <div class="setting-row">
        <div class="setting-label">
          <span class="setting-name">LLM claim extraction</span>
          <span class="setting-desc">When ON, an LLM extracts and maps each claim to its sources. When OFF, a faster heuristic is used — trust signals stay computable but may be less accurate.</span>
        </div>
        <div class="setting-control">
          <label class="toggle">
            <input type="checkbox" id="setting-fact-extraction" ${settings.fact_extraction_enabled ? 'checked' : ''}/>
            <span class="toggle-slider"></span>
          </label>
        </div>
      </div>

      <div class="explainer-h3">Data</div>

      <div class="setting-row">
        <div class="setting-label">
          <span class="setting-name">Clear answer cache</span>
          <span class="setting-desc">Wipes <code>cache.json</code>. Your chat history is not affected.</span>
        </div>
        <div class="setting-control">
          <button class="setting-action-btn danger" id="setting-clear-cache" type="button">Clear cache</button>
        </div>
      </div>

      <div class="setting-row">
        <div class="setting-label">
          <span class="setting-name">Reset settings</span>
          <span class="setting-desc">Restore every setting above to its default value.</span>
        </div>
        <div class="setting-control">
          <button class="setting-action-btn" id="setting-reset" type="button">Reset to defaults</button>
        </div>
      </div>

      <div id="setting-toast" class="setting-toast" hidden></div>
    </div>
  `;

  modal.querySelector('.modal-close').addEventListener('click', closeSettings);

  const cacheInput = modal.querySelector('#setting-cache-ttl');
  cacheInput.addEventListener('change', async () => {
    const hours = Math.max(0, Math.min(720, parseInt(cacheInput.value, 10) || 0));
    cacheInput.value = hours;
    const result = await patchSettings({ cache_ttl_seconds: hours * 3600 });
    showSettingsToast(modal, result ? `Cache TTL set to ${hours}h.` : 'Failed to save', result ? 'success' : 'error');
  });

  const historyInput = modal.querySelector('#setting-history');
  historyInput.addEventListener('change', async () => {
    const limit = Math.max(0, Math.min(20, parseInt(historyInput.value, 10) || 0));
    historyInput.value = limit;
    const result = await patchSettings({ history_turn_limit: limit });
    showSettingsToast(modal, result ? `History limit set to ${limit}.` : 'Failed to save', result ? 'success' : 'error');
  });

  const verifyToggle = modal.querySelector('#setting-verification');
  verifyToggle.addEventListener('change', async () => {
    const result = await patchSettings({ verification_enabled: verifyToggle.checked });
    showSettingsToast(modal, result ? `Verification ${verifyToggle.checked ? 'enabled' : 'disabled'}.` : 'Failed to save', result ? 'success' : 'error');
  });

  const factToggle = modal.querySelector('#setting-fact-extraction');
  factToggle.addEventListener('change', async () => {
    const result = await patchSettings({ fact_extraction_enabled: factToggle.checked });
    showSettingsToast(modal, result ? `LLM claim extraction ${factToggle.checked ? 'enabled' : 'disabled'}.` : 'Failed to save', result ? 'success' : 'error');
  });

  modal.querySelector('#setting-clear-cache').addEventListener('click', async () => {
    if (!confirm('Wipe the entire answer cache? This cannot be undone.')) return;
    const result = await clearAnswerCache();
    showSettingsToast(modal, result ? 'Cache cleared.' : 'Failed to clear cache.', result ? 'success' : 'error');
  });

  modal.querySelector('#setting-reset').addEventListener('click', async () => {
    if (!confirm('Reset all settings to their defaults?')) return;
    const data = await resetSettings();
    if (!data) {
      showSettingsToast(modal, 'Failed to reset.', 'error');
      return;
    }
    const s = data.settings;
    cacheInput.value = Math.round((s.cache_ttl_seconds || 0) / 3600);
    historyInput.value = s.history_turn_limit;
    verifyToggle.checked = !!s.verification_enabled;
    factToggle.checked = !!s.fact_extraction_enabled;
    showSettingsToast(modal, 'Settings reset to defaults.', 'success');
  });

  backdrop.appendChild(modal);
  return backdrop;
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
    agentBlock.appendChild(
      buildSection(
        'Trust signals',
        null,
        buildTrustGrid(turn.trust_signals),
        buildTrustActions(turn.trust_signals, turn.sources || [], turn.facts || [])
      )
    );
  }

  turnEl.appendChild(agentBlock);
  conversation.appendChild(turnEl);
}

/* ----------------- streaming ----------------- */

function startResearch(question, options) {
  const opts = options || {};
  const isRegenerate = !!opts.regenerate;
  isAsking = true;
  setSubmitMode('stop');
  if (!isRegenerate) {
    input.value = '';
    autosizeInput();
  }
  removeAllRegenerateButtons();
  setStatus('thinking', isRegenerate ? 'Regenerating' : 'Researching');
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
  if (isRegenerate) params.set('regenerate', 'true');
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

  es.addEventListener('token', (event) => {
    const data = JSON.parse(event.data);
    if (!data.text) return;
    if (traceLabel) traceLabel.textContent = 'Drafting answer…';
    let answerCard = agentBlock.querySelector('.answer-card');
    if (!answerCard) {
      answerCard = buildAnswerCard('');
      agentBlock.appendChild(answerCard);
    }
    if (answerCard._streamedText == null) answerCard._streamedText = '';
    answerCard._streamedText += data.text;
    if (window.marked) {
      answerCard.innerHTML = marked.parse(answerCard._streamedText);
    } else {
      answerCard.textContent = answerCard._streamedText;
    }
    scrollToBottom();
  });

  es.addEventListener('tool_call', (event) => {
    const data = JSON.parse(event.data);
    // Clear any intermediate reasoning tokens — they're pre-tool noise.
    const answerCard = agentBlock.querySelector('.answer-card');
    if (answerCard) {
      answerCard._streamedText = '';
      answerCard.innerHTML = '';
    }
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
    // Sync the streamed-text accumulator with the authoritative answer
    answerCard._streamedText = data.text || '';
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
      agentBlock.appendChild(
        buildSection(
          'Trust signals',
          null,
          buildTrustGrid(data.trust_signals),
          buildTrustActions(data.trust_signals, data.sources || [], data.facts || [])
        )
      );
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
    // Stamp a regenerate button on the just-completed turn.
    removeAllRegenerateButtons();
    agentBlock.appendChild(buildRegenerateButton(question));
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
