// DOM Elements
const fileInput = document.getElementById('pdfUpload');
const fileNameDisplay = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');

const chatBox = document.getElementById('chatBox');
const questionInput = document.getElementById('questionInput');
const askBtn = document.getElementById('askBtn');

// Eval DOM
const evalQuestion    = document.getElementById('evalQuestion');
const evalGroundTruth = document.getElementById('evalGroundTruth');
const addTestCaseBtn  = document.getElementById('addTestCaseBtn');
const testCaseList    = document.getElementById('testCaseList');
const runEvalBtn      = document.getElementById('runEvalBtn');
const evalStatus      = document.getElementById('evalStatus');
const evalDashboard   = document.getElementById('evalDashboard');

// State
let conversationHistory = [];
let testCases = []; // [{question, ground_truth}]

// ─── Security Helper ─────────────────────────────────────────────────────────
// Escapes HTML characters to prevent Cross-Site Scripting (XSS) when using innerHTML
function escapeHTML(str) {
    if (!str) return '';
    return str.toString()
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// ─── Error Parsing Helper ────────────────────────────────────────────────────
// Extracts a clean, human-readable message from any API error response.
// Handles: FastAPI HTTPException, slowapi rate limit, and network failures.
function parseApiError(response, data) {
    if (response.status === 429) {
        return '⏳ Too many requests. Please wait a moment before trying again.';
    }
    if (response.status === 413) {
        return '📁 File too large. Please upload a PDF under 50MB.';
    }
    if (response.status === 400) {
        // Our structured error: { detail: { error: "..." } }
        return data?.detail?.error || data?.detail || 'Invalid request. Please check your input.';
    }
    if (response.status === 500) {
        return '🔧 Server error. Please try again in a moment.';
    }
    // Catch-all: extract whatever message is available
    if (data?.detail?.error) return data.detail.error;
    if (typeof data?.detail === 'string') return data.detail;
    return 'Something went wrong. Please try again.';
}

// ─── 1. File Upload Logic ────────────────────────────────────────────────────
fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) {
        fileNameDisplay.textContent = 'Select PDF Document(s)...';
        uploadBtn.disabled = true;
        return;
    }

    // Show file count or single file name
    fileNameDisplay.textContent = files.length === 1
        ? files[0].name
        : `${files.length} files selected`;
    uploadBtn.disabled = false;

    // Warn immediately if any file exceeds the size limit
    const oversized = files.filter(f => f.size > 50 * 1024 * 1024);
    if (oversized.length > 0) {
        setUploadStatus('error', `📁 ${oversized.map(f => f.name).join(', ')} exceed the 50MB limit.`);
        uploadBtn.disabled = true;
    } else {
        uploadStatus.classList.add('hidden');
    }
});

uploadBtn.addEventListener('click', async () => {
    const files = Array.from(fileInput.files);
    if (files.length === 0) return;

    // Validate all files before starting any upload
    const oversized = files.filter(f => f.size > 50 * 1024 * 1024);
    if (oversized.length > 0) {
        setUploadStatus('error', `📁 ${oversized.map(f => f.name).join(', ')} exceed the 50MB limit.`);
        return;
    }

    uploadBtn.disabled = true;
    let succeeded = 0;
    let failed = 0;

    // Process files sequentially — one at a time to avoid overloading the server
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setUploadStatus('loading', `⚙️ Ingesting file ${i + 1}/${files.length}: ${file.name}...`);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/ingest', { method: 'POST', body: formData });

            let data = {};
            try { data = await response.json(); } catch { /* non-JSON body */ }

            if (response.ok) {
                succeeded++;
                // Show running tally after each success
                setUploadStatus('loading', `✅ ${file.name} done (${succeeded}/${files.length})`);
            } else {
                failed++;
                // Show error for this file but continue with the rest
                setUploadStatus('loading',
                    `⚠️ ${file.name} failed: ${parseApiError(response, data)} — continuing...`);
                await new Promise(r => setTimeout(r, 2000)); // Pause briefly so user can read the error
            }
        } catch {
            failed++;
            setUploadStatus('loading', `🌐 ${file.name}: Cannot reach server. Skipping...`);
            await new Promise(r => setTimeout(r, 2000));
        }
    }

    // Final summary after all files are processed
    if (failed === 0) {
        setUploadStatus('success', `✅ All ${succeeded} document(s) ingested successfully!`);
    } else if (succeeded > 0) {
        setUploadStatus('error', `⚠️ ${succeeded} succeeded, ${failed} failed. Check errors above.`);
    } else {
        setUploadStatus('error', `❌ All ${failed} document(s) failed to ingest.`);
    }

    uploadBtn.disabled = false;
});

function setUploadStatus(type, message) {
    uploadStatus.classList.remove('hidden', 'status-loading', 'status-success', 'status-error');
    const classMap = { loading: 'status-loading', success: 'status-success', error: 'status-error' };
    uploadStatus.classList.add(classMap[type]);
    uploadStatus.textContent = message;
}

// ─── 2. Chat Logic ────────────────────────────────────────────────────────────
async function submitQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    appendMessage('user', question);
    questionInput.value = '';
    askBtn.disabled = true;

    // Add a temporary animated thinking indicator
    const thinkingId = appendMessage('ai', '__typing__');

    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, conversation_history: conversationHistory }),
        });

        const data = await response.json();

        if (response.ok) {
            updateMessage(thinkingId, data.answer, 'answer');

            // Update conversation memory
            conversationHistory.push({ role: 'user', content: question });
            conversationHistory.push({ role: 'assistant', content: data.answer });

            // Safety valve: cap at last 10 exchanges
            if (conversationHistory.length > 20) {
                conversationHistory = conversationHistory.slice(-20);
            }
        } else {
            // Show the clean parsed error inside the chat bubble
            updateMessage(thinkingId, parseApiError(response, data), 'error');
        }
    } catch (err) {
        updateMessage(thinkingId, '🌐 Cannot reach the server. Is it running?', 'error');
    } finally {
        askBtn.disabled = false;
    }
}

questionInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') submitQuestion();
});
askBtn.addEventListener('click', submitQuestion);

// ─── 3. UI Helper Functions ───────────────────────────────────────────────────
function appendMessage(role, text) {
    const id = 'msg-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    msgDiv.id = id;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'ai' ? 'AI' : 'You';

    const content = document.createElement('div');
    content.className = 'msg-content';

    if (text === '__typing__') {
        content.innerHTML = `<span class="typing-dot-1"></span><span class="typing-dot-2"></span><span class="typing-dot-3"></span>`;
    } else {
        content.textContent = text;
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(content);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    return id;
}

function updateMessage(id, newText, type = 'answer') {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;

    const contentDiv = msgDiv.querySelector('.msg-content');
    contentDiv.textContent = newText;

    if (type === 'error') {
        contentDiv.classList.add('is-error');
    }
}


// ─── TAB SWITCHING ───────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;

        // Update tab button active state
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Show/hide sidebar content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
        document.getElementById(`tab-${tab}`).classList.remove('hidden');

        // Show/hide main area view
        if (tab === 'upload') {
            document.getElementById('chatView').classList.remove('hidden');
            document.getElementById('evalView').classList.add('hidden');
        } else {
            document.getElementById('chatView').classList.add('hidden');
            document.getElementById('evalView').classList.remove('hidden');
        }
    });
});


// ─── EVALUATION: ADD / REMOVE TEST CASES ──────────────────────────────────────
addTestCaseBtn.addEventListener('click', () => {
    const q  = evalQuestion.value.trim();
    const gt = evalGroundTruth.value.trim();
    if (!q || !gt) {
        setEvalStatus('error', 'Please fill in both the question and the ground truth.');
        return;
    }
    if (testCases.length >= 15) {
        setEvalStatus('error', 'Maximum 15 test cases allowed per run.');
        return;
    }

    testCases.push({ question: q, ground_truth: gt });
    evalQuestion.value    = '';
    evalGroundTruth.value = '';
    evalStatus.classList.add('hidden');
    renderTestCaseList();
    runEvalBtn.disabled = testCases.length === 0;
});

function renderTestCaseList() {
    testCaseList.innerHTML = '';
    testCases.forEach((tc, i) => {
        const item = document.createElement('div');
        item.className = 'test-case-item';
        // XSS Prevention: Escape user input before interpolating into innerHTML
        const safeQuestion = escapeHTML(tc.question);
        item.innerHTML = `
            <span class="tc-question" title="${safeQuestion}">Q${i+1}: ${safeQuestion}</span>
            <button class="tc-remove" data-index="${i}" title="Remove">×</button>
        `;
        testCaseList.appendChild(item);
    });

    // Wire up remove buttons
    testCaseList.querySelectorAll('.tc-remove').forEach(btn => {
        btn.addEventListener('click', () => {
            testCases.splice(parseInt(btn.dataset.index), 1);
            renderTestCaseList();
            runEvalBtn.disabled = testCases.length === 0;
        });
    });
}


// ─── EVALUATION: RUN ───────────────────────────────────────────────────────────────
runEvalBtn.addEventListener('click', async () => {
    if (testCases.length === 0) return;

    runEvalBtn.disabled = true;
    setEvalStatus('loading', `⏳ Running pipeline on ${testCases.length} question(s) then scoring with Ragas... (2–4 min)`);

    try {
        const response = await fetch('/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ test_cases: testCases }),
        });

        let data = {};
        try { data = await response.json(); } catch { }

        if (response.ok) {
            setEvalStatus('success', `✅ Evaluation complete!`);
            renderEvalResults(data);
        } else {
            setEvalStatus('error', parseApiError(response, data));
        }
    } catch {
        setEvalStatus('error', '🌐 Cannot reach the server.');
    } finally {
        runEvalBtn.disabled = false;
    }
});

function setEvalStatus(type, message) {
    evalStatus.classList.remove('hidden', 'status-loading', 'status-success', 'status-error');
    const classMap = { loading: 'status-loading', success: 'status-success', error: 'status-error' };
    evalStatus.classList.add(classMap[type]);
    evalStatus.textContent = message;
}


// ─── EVALUATION: RENDER RESULTS ────────────────────────────────────────────────────
function scoreClass(v) {
    if (v === null || v === undefined) return '';
    if (v >= 0.85) return 'score-excellent s-ex';
    if (v >= 0.65) return 'score-good s-gd';
    return 'score-poor s-pr';
}

function tierClass(v) {
    if (v === null || v === undefined) return '';
    if (v >= 0.85) return 'tier-excellent';
    if (v >= 0.65) return 'tier-good';
    return 'tier-poor';
}

function fmt(v) {
    return v !== null && v !== undefined ? v.toFixed(3) : '—';
}

function renderEvalResults(data) {
    const { aggregate, per_question } = data;

    const METRICS = [
        { key: 'faithfulness',      label: 'Faithfulness',      icon: '🛡️', desc: 'Hallucination prevention' },
        { key: 'answer_relevancy',  label: 'Answer Relevancy',  icon: '🎯', desc: 'On-topic answers' },
        { key: 'context_precision', label: 'Context Precision', icon: '🔍', desc: 'Retrieval quality' },
        { key: 'context_recall',    label: 'Context Recall',    icon: '📚', desc: 'Retrieval completeness' },
    ];

    // ── Metric cards ─────────────────────────────────────────────────────────
    let cardsHTML = '<div class="metric-grid">';
    METRICS.forEach(m => {
        const v = aggregate[m.key];
        cardsHTML += `
            <div class="metric-card ${tierClass(v)}">
                <div class="metric-icon">${m.icon}</div>
                <div class="metric-label">${m.label}</div>
                <div class="metric-score">${fmt(v)}</div>
                <div class="metric-desc">${m.desc}</div>
            </div>`;
    });
    cardsHTML += '</div>';

    // ── Per-question table ────────────────────────────────────────────────────
    let tableHTML = `
        <div class="pq-card glass">
            <div class="pq-title">Per-Question Breakdown</div>
            <table class="pq-table">
                <thead><tr>
                    <th>Question</th>
                    <th>Faith.</th><th>Relevancy</th><th>Precision</th><th>Recall</th>
                </tr></thead>
                <tbody>`;

    per_question.forEach(row => {
        const sc = (v) => {
            const cls = v >= 0.85 ? 's-ex' : v >= 0.65 ? 's-gd' : 's-pr';
            return `<td class="sc ${cls}">${fmt(v)}</td>`;
        };
        // XSS Prevention: Escape user input before interpolating into innerHTML
        const safeQuestion = escapeHTML(row.question);
        tableHTML += `<tr>
            <td><span class="q-text" title="${safeQuestion}">${safeQuestion}</span></td>
            ${sc(row.faithfulness)}${sc(row.answer_relevancy)}${sc(row.context_precision)}${sc(row.context_recall)}
        </tr>`;
    });

    tableHTML += '</tbody></table></div>';

    evalDashboard.innerHTML = cardsHTML + tableHTML;
}


