// ============================================================
// VOICE INPUT — Web Speech API (Browser Native)
// No API key required • Works in Chrome, Edge, Safari 17+
// ============================================================

(function () {
    'use strict';

    // ── Browser support check ────────────────────────────────
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        console.warn('[Voice] SpeechRecognition API not available in this browser.');
        // Show a subtle badge on the mic button
        const micBtn = document.getElementById('micBtn');
        if (micBtn) {
            micBtn.title = 'Voice input not supported in this browser. Use Chrome or Edge.';
            micBtn.style.opacity = '0.4';
            micBtn.style.cursor = 'not-allowed';
            micBtn.onclick = () => {
                showVoiceStatus('⚠️ Voice input requires Chrome or Edge browser.', 'error');
            };
        }
        return;
    }

    // ── State ────────────────────────────────────────────────
    let recognition = null;
    let isListening = false;
    let silenceTimer = null;

    // ── DOM refs ─────────────────────────────────────────────
    function getMicBtn() { return document.getElementById('micBtn'); }
    function getInput()  { return document.getElementById('queryInput'); }
    function getStatus() { return document.getElementById('voiceStatus'); }

    // ── Status display ───────────────────────────────────────
    function showVoiceStatus(msg, type = 'info') {
        const el = getStatus();
        if (!el) return;
        el.textContent = msg;
        el.style.display = 'block';
        el.className = 'voice-status voice-status--' + type;
        if (type !== 'recording') {
            clearTimeout(silenceTimer);
            silenceTimer = setTimeout(() => {
                el.style.display = 'none';
            }, 3000);
        }
    }

    // ── Build recognition instance ───────────────────────────
    function buildRecognition() {
        const r = new SpeechRecognition();
        r.continuous = false;
        r.interimResults = true;
        r.lang = 'en-US';
        r.maxAlternatives = 1;

        r.onstart = () => {
            isListening = true;
            const btn = getMicBtn();
            if (btn) {
                btn.classList.add('recording');
                btn.querySelector('i').className = 'fas fa-stop';
                btn.title = 'Click to stop recording';
            }
            showVoiceStatus('🎤 Listening… speak now', 'recording');
        };

        r.onresult = (event) => {
            let interim = '';
            let final = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const t = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    final += t;
                } else {
                    interim += t;
                }
            }

            const input = getInput();
            if (input) {
                input.value = final || interim;
            }

            if (final) {
                showVoiceStatus('✅ Transcribed — sending…', 'success');
                stopListening();
                // Auto-send after a short delay
                setTimeout(() => {
                    if (typeof sendQuery === 'function' && final.trim()) {
                        sendQuery();
                    }
                }, 400);
            }
        };

        r.onerror = (event) => {
            console.error('[Voice] Error:', event.error);
            const messages = {
                'not-allowed': '🚫 Microphone permission denied. Please allow access.',
                'no-speech': '🔇 No speech detected. Try again.',
                'network': '🌐 Network error. Check connection.',
                'aborted': '',
            };
            const msg = messages[event.error] || `❌ Voice error: ${event.error}`;
            if (msg) showVoiceStatus(msg, 'error');
            stopListening(false);
        };

        r.onend = () => {
            stopListening(false);
        };

        return r;
    }

    // ── Start/Stop ───────────────────────────────────────────
    function startListening() {
        if (isListening) return;
        recognition = buildRecognition();
        try {
            recognition.start();
        } catch (e) {
            showVoiceStatus('❌ Could not start voice input.', 'error');
        }
    }

    function stopListening(showMsg = true) {
        isListening = false;
        if (recognition) {
            try { recognition.stop(); } catch (e) { /* ignore */ }
            recognition = null;
        }
        const btn = getMicBtn();
        if (btn) {
            btn.classList.remove('recording');
            btn.querySelector('i').className = 'fas fa-microphone';
            btn.title = 'Voice input (Ctrl+M)';
        }
        if (showMsg) showVoiceStatus('');
    }

    // ── Public toggle function (called from HTML onclick) ────
    window.toggleMic = function () {
        if (isListening) {
            stopListening();
            showVoiceStatus('⏹ Recording stopped.', 'info');
        } else {
            startListening();
        }
    };

    // ── Keyboard shortcut: Ctrl+M ────────────────────────────
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'm') {
            e.preventDefault();
            window.toggleMic();
        }
    });

    console.log('[Voice] Web Speech API initialized ✅');
})();
