// ============================================================
// AGENTS.JS — Agent Management & Per-Agent Operations
// ============================================================

(function () {
    'use strict';

    // ── State ─────────────────────────────────────────────────
    let activeAgentId   = null;
    let activeAgentName = null;
    let agentsList      = [];

    // ── Public getters used by app.js ─────────────────────────
    window.getActiveAgentId   = () => activeAgentId;
    window.getActiveAgentName = () => activeAgentName;

    // ── Init: called when page loads ──────────────────────────
    window.initAgents = async function () {
        await loadAgents();
    };

    // Expose globally so agents.html and index.html can both call it
    window.reloadAgents = loadAgents;

    // ── Load agents from backend ──────────────────────────────
    async function loadAgents() {
        console.log('[Agents] Fetching agents from /agents ...');
        try {
            const res = await authFetch('/agents');
            if (!res.ok) {
                console.error('[Agents] /agents returned', res.status);
                agentsList = [];
            } else {
                const data = await res.json();
                agentsList = Array.isArray(data) ? data : [];
                console.log('[Agents] Agents loaded:');
                console.table(agentsList);
            }
        } catch (e) {
            console.error('[Agents] Failed to load agents:', e);
            agentsList = [];
        }
        // Update the sidebar dropdown (if on index page)
        populateAgentDropdown();
        
        // Auto-select if none selected
        if (!activeAgentId && agentsList.length > 0) {
            autoSelectFirstAgent();
        }
    }

    function autoSelectFirstAgent() {
        if (agentsList.length > 0) {
            const first = agentsList[0];
            console.log('[Agents] Auto-selecting first available agent:', first.name);
            window.selectAgent(first.id, first.name);
        }
    }

    // ── Populate the Agent Selector dropdown ──────────────────
    function populateAgentDropdown() {
        const selectEl = document.getElementById('agentSelector');
        if (!selectEl) {
            console.log('[Agents] No #agentSelector found on this page, skipping dropdown.');
            return;
        }

        console.log('[Agents] Populating dropdown with', agentsList.length, 'agent(s)');

        if (agentsList.length === 0) {
            selectEl.innerHTML = '<option value="" disabled selected>No agents yet — create one on the Agents page!</option>';
            selectEl.disabled = true;
            return;
        }

        selectEl.disabled = false;
        selectEl.innerHTML = '<option value="" disabled selected>— Select an Agent —</option>' +
            agentsList.map(agent =>
                `<option value="${agent.id}" ${agent.id === activeAgentId ? 'selected' : ''}>${escapeHtml(agent.name)}</option>`
            ).join('');

        // If there is an already-selected agent, keep it selected
        if (activeAgentId) {
            selectEl.value = activeAgentId;
        } else {
            // If no agent selected but we have some, default to first or keep 'Select'
            if (selectEl.options.length > 1 && !selectEl.value) {
                // Let handleAgentSelectionChange take over if needed
            }
        }
    }

    // ── Global Event Delegation ───────────────────────────────
    // This is more robust than inline onchange
    document.addEventListener('change', (e) => {
        if (e.target && e.target.id === 'agentSelector') {
            const agentId = e.target.value;
            console.log('[Agents] Dropdown changed to:', agentId);
            window.handleAgentSelectionChange(agentId);
        }
    });

    // ── Handle dropdown change from HTML ──────────────────────
    window.handleAgentSelectionChange = function (agentId) {
        if (!agentId) return;
        const selected = agentsList.find(a => a.id === agentId);
        if (selected) window.selectAgent(selected.id, selected.name);
    };

    // ── Select / activate an agent ────────────────────────────
    window.selectAgent = function (agentId, agentName) {
        console.log('[Agents] Selecting agent:', agentId, agentName);
        activeAgentId   = agentId;
        activeAgentName = agentName;

        updateAgentBadge(agentName);
        populateAgentDropdown();
        
        if (typeof updateAgentUploadBtnState === 'function') {
            updateAgentUploadBtnState();
        }

        const hint = document.getElementById('agentUploadHint');
        if (hint) hint.style.display = agentId ? 'none' : 'block';

        if (activeAgentId) {
            loadAgentDocuments(activeAgentId);
        } else {
            clearAgentDocuments();
        }
    };

    // ── Header badge ──────────────────────────────────────────
    function updateAgentBadge(name) {
        const badge = document.getElementById('activeAgentBadge');
        if (!badge) return;
        if (name) {
            badge.textContent = `🤖 ${name}`;
            badge.style.display = 'inline-flex';
        } else {
            badge.style.display = 'none';
        }
    }

    // ── Create new agent (called from index.html modal) ───────
    window.submitCreateAgent = async function () {
        const nameEl = document.getElementById('newAgentName');
        const descEl = document.getElementById('newAgentDesc');
        const errEl  = document.getElementById('createAgentError');
        const btn    = document.getElementById('createAgentBtn');

        const name = nameEl?.value.trim() || '';
        const desc = descEl?.value.trim() || '';

        if (!name) {
            if (errEl) { errEl.textContent = 'Agent name is required.'; errEl.style.display = 'block'; }
            return;
        }

        if (errEl) errEl.style.display = 'none';
        if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Creating…'; }

        try {
            const res = await authFetch('/agents', {
                method: 'POST',
                body: JSON.stringify({ name, description: desc })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Failed to create agent');
            }

            const agent = await res.json();
            agentsList.push(agent);
            populateAgentDropdown();
            closeCreateAgentModal();

            // Auto-select the newly created agent
            window.selectAgent(agent.id, agent.name);

            if (nameEl) nameEl.value = '';
            if (descEl) descEl.value = '';
        } catch (e) {
            if (errEl) { errEl.textContent = e.message; errEl.style.display = 'block'; }
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-plus"></i> Create Agent'; }
        }
    };

    // ── Upload documents to active agent ─────────────────────
    window.uploadToActiveAgent = async function () {
        if (!activeAgentId) {
            alert('Please select an agent first.');
            return;
        }

        const fileInput = document.getElementById('agentFileInput');
        if (!fileInput || !fileInput.files.length) {
            alert('Please choose files to upload.');
            return;
        }

        const statusEl = document.getElementById('agentUploadStatus');
        const btn      = document.getElementById('agentUploadBtn');

        if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Uploading…'; }
        if (statusEl) { statusEl.textContent = '📤 Uploading and indexing…'; statusEl.className = 'upload-status-text info'; }

        const formData = new FormData();
        for (const f of fileInput.files) formData.append('files', f);
        formData.append('agent_id', activeAgentId); // Use unified /upload with agent_id

        try {
            const res = await authFetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Upload failed');

            const ok  = data.files_processed || 0;
            const bad = data.files_failed    || 0;
            let msg = `✅ ${ok} file(s) indexed into "${activeAgentName}"`;
            if (bad > 0) msg += ` (${bad} failed)`;

            if (statusEl) { statusEl.textContent = msg; statusEl.className = 'upload-status-text success'; }
            fileInput.value = '';
            renderAgentFileList([]);
            loadAgentDocuments(activeAgentId);
        } catch (e) {
            if (statusEl) { statusEl.textContent = `❌ ${e.message}`; statusEl.className = 'upload-status-text error'; }
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fas fa-bolt"></i> Index to Agent'; }
        }
    };

    // ── Agent file input change ───────────────────────────────
    window.handleAgentFileSelect = function (event) {
        const files = Array.from(event.target.files || []);
        console.log('[Agents] Requested files for upload:', files.map(f => f.name));
        renderAgentFileList(files);
        updateAgentUploadBtnState();
    };

    function updateAgentUploadBtnState() {
        const fileInput = document.getElementById('agentFileInput');
        const btn = document.getElementById('agentUploadBtn');
        if (!btn) return;

        const hasFiles = fileInput && fileInput.files && fileInput.files.length > 0;
        const hasAgent = !!activeAgentId;

        console.log('[Agents] Button state check:', { hasFiles, hasAgent, activeAgentId });
        btn.disabled = !hasFiles || !hasAgent;
        
        if (btn.disabled) {
            btn.title = !hasAgent ? 'Select an agent first' : 'Select files first';
        } else {
            btn.title = 'Ready to index';
        }
    }

    function renderAgentFileList(files) {
        const container = document.getElementById('agentFileList');
        if (!container) return;
        if (!files.length) { container.innerHTML = ''; return; }
        container.innerHTML = files.map(f =>
            `<div class="file-pill"><i class="fas fa-file"></i> ${escapeHtml(f.name)}</div>`
        ).join('');
    }

    // ── Load documents indexed for given agent ────────────────
    async function loadAgentDocuments(agentId) {
        try {
            const res  = await authFetch(`/agents/${agentId}/documents`);
            const data = await res.json();
            renderAgentDocuments(Array.isArray(data) ? data : []);
        } catch (e) {
            console.error('[Agents] loadAgentDocuments error:', e);
        }
    }

    function renderAgentDocuments(docs) {
        const container = document.getElementById('agentDocsContainer');
        if (!container) return;
        if (!docs.length) {
            container.innerHTML = '<p class="no-docs-msg">No documents indexed yet.</p>';
            return;
        }
        container.innerHTML = docs.map(d => `
            <div class="doc-pill">
                <i class="fas fa-file-lines"></i>
                <span>${escapeHtml(d.file_name)}</span>
            </div>`
        ).join('');
    }

    function clearAgentDocuments() {
        const container = document.getElementById('agentDocsContainer');
        if (container) container.innerHTML = '';
    }

    // ── Modal helpers ─────────────────────────────────────────
    window.openCreateAgentModal = function () {
        const modal = document.getElementById('createAgentModal');
        if (modal) modal.style.display = 'flex';
    };

    window.closeCreateAgentModal = function () {
        const modal = document.getElementById('createAgentModal');
        if (modal) modal.style.display = 'none';
        const errEl = document.getElementById('createAgentError');
        if (errEl) errEl.style.display = 'none';
    };

    document.addEventListener('click', (e) => {
        const modal = document.getElementById('createAgentModal');
        if (modal && e.target === modal) closeCreateAgentModal();
    });

    // ── Escape helpers ────────────────────────────────────────
    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    console.log('[Agents] Agent manager initialized ✅');
})();
