// ============================================================
// DOCINTTEL AI - ULTRA PREMIUM FRONTEND V2
// Maximum Interactivity & Visual Effects
// ============================================================

// State
let selectedFiles = [];

// ============================================================
// RESET CHAT
// ============================================================

async function resetChat() {
    // Clear chat container except welcome message
    const container = document.getElementById('chatContainer');
    const welcomeMsg = document.getElementById('welcomeMessage');

    // Get all messages (not welcome)
    const messages = container.querySelectorAll('.message');

    // Animate messages out
    messages.forEach((msg, index) => {
        setTimeout(() => {
            msg.style.transition = 'all 0.3s ease';
            msg.style.opacity = '0';
            msg.style.transform = 'translateX(-50px)';
            setTimeout(() => msg.remove(), 300);
        }, index * 50);
    });

    // Wait for animations
    await new Promise(resolve => setTimeout(resolve, messages.length * 50 + 400));

    // Show welcome message again
    if (welcomeMsg) {
        welcomeMsg.style.display = 'flex';
        welcomeMsg.style.opacity = '0';
        welcomeMsg.style.transform = 'translateY(30px) scale(0.95)';

        setTimeout(() => {
            welcomeMsg.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
            welcomeMsg.style.opacity = '1';
            welcomeMsg.style.transform = 'translateY(0) scale(1)';
        }, 50);
    }

    // Clear input
    document.getElementById('queryInput').value = '';
    document.getElementById('queryInput').focus();

    // Optional: Clear server history
    try {
        await fetch('/history/clear', { method: 'POST' });
    } catch (e) {
        // Server might not have this endpoint, that's okay
    }

    // Show notification
    showNotification('Chat cleared! Ready for a new conversation.', 'success');
}

// Show notification toast
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    notification.style.cssText = `
        position: fixed;
        bottom: 100px;
        left: 50%;
        transform: translateX(-50%);
        background: ${type === 'success' ? 'rgba(16, 185, 129, 0.9)' : 'rgba(139, 92, 246, 0.9)'};
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 500;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        z-index: 10000;
        animation: notificationSlide 0.4s ease;
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(-50%) translateY(20px)';
        notification.style.transition = 'all 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 2500);
}

// Add notification animation
const notifStyle = document.createElement('style');
notifStyle.textContent = `
    @keyframes notificationSlide {
        from { opacity: 0; transform: translateX(-50%) translateY(20px); }
        to { opacity: 1; transform: translateX(-50%) translateY(0); }
    }
`;
document.head.appendChild(notifStyle);

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    setupFileInput();
    setupDragDrop();
    setupCursorGlow();
    createParticles();
    setupInteractiveEffects();

    // Focus input on page load with delay for animation
    setTimeout(() => {
        document.getElementById('queryInput').focus();
    }, 500);
});

// ============================================================
// CURSOR GLOW EFFECT
// ============================================================

function setupCursorGlow() {
    const cursorGlow = document.getElementById('cursorGlow');
    if (!cursorGlow) return;

    let mouseX = 0, mouseY = 0;
    let glowX = 0, glowY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    // Smooth follow animation
    function animateGlow() {
        const speed = 0.15;
        glowX += (mouseX - glowX) * speed;
        glowY += (mouseY - glowY) * speed;

        cursorGlow.style.left = glowX + 'px';
        cursorGlow.style.top = glowY + 'px';

        requestAnimationFrame(animateGlow);
    }

    animateGlow();

    // Hide when mouse leaves window
    document.addEventListener('mouseleave', () => {
        cursorGlow.style.opacity = '0';
    });

    document.addEventListener('mouseenter', () => {
        cursorGlow.style.opacity = '1';
    });
}

// ============================================================
// FLOATING PARTICLES
// ============================================================

function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;

    const particleCount = 30;
    const colors = ['#8b5cf6', '#6366f1', '#22d3ee', '#f472b6', '#34d399'];

    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';

        // Random properties
        const size = Math.random() * 4 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const delay = Math.random() * 10;
        const duration = Math.random() * 10 + 10;

        particle.style.cssText = `
            width: ${size}px;
            height: ${size}px;
            background: ${color};
            left: ${left}%;
            animation-delay: ${delay}s;
            animation-duration: ${duration}s;
            box-shadow: 0 0 ${size * 2}px ${color};
        `;

        container.appendChild(particle);
    }
}

// ============================================================
// INTERACTIVE EFFECTS
// ============================================================

function setupInteractiveEffects() {
    // Add ripple effect to buttons
    document.querySelectorAll('button, .prompt-btn, .btn-upload').forEach(btn => {
        btn.addEventListener('click', createRipple);
    });

    // Add tilt effect to sidebar sections
    document.querySelectorAll('.sidebar-section').forEach(section => {
        section.addEventListener('mousemove', handleTilt);
        section.addEventListener('mouseleave', resetTilt);
    });

    // Input focus animation
    const input = document.getElementById('queryInput');
    if (input) {
        input.addEventListener('focus', () => {
            input.parentElement.classList.add('input-focused');
        });
        input.addEventListener('blur', () => {
            input.parentElement.classList.remove('input-focused');
        });
    }
}

function createRipple(e) {
    const button = e.currentTarget;
    const rect = button.getBoundingClientRect();

    const ripple = document.createElement('span');
    ripple.className = 'ripple-effect';
    ripple.style.cssText = `
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.4);
        transform: scale(0);
        animation: ripple 0.6s ease-out;
        pointer-events: none;
        left: ${e.clientX - rect.left}px;
        top: ${e.clientY - rect.top}px;
        width: 100px;
        height: 100px;
        margin-left: -50px;
        margin-top: -50px;
    `;

    button.style.position = 'relative';
    button.style.overflow = 'hidden';
    button.appendChild(ripple);

    setTimeout(() => ripple.remove(), 600);
}

// Add ripple keyframes dynamically
const rippleStyle = document.createElement('style');
rippleStyle.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(rippleStyle);

function handleTilt(e) {
    const section = e.currentTarget;
    const rect = section.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rotateX = (y - centerY) / 20;
    const rotateY = (centerX - x) / 20;

    section.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-3px)`;
}

function resetTilt(e) {
    e.currentTarget.style.transform = '';
}

// ============================================================
// CHAT FUNCTIONS
// ============================================================

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuery();
    }
}

async function sendQuery() {
    const input = document.getElementById('queryInput');
    const question = input.value.trim();

    if (!question) return;

    // Add send animation
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.classList.add('sending');
    setTimeout(() => sendBtn.classList.remove('sending'), 300);

    // Clear input
    input.value = '';

    // Hide welcome message
    hideWelcome();

    // Add user message
    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
    addMessage('user', question, timestamp);

    // Play send sound (optional)
    playSound('send');

    // Show typing indicator
    showTyping();

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            throw new Error('Query failed');
        }

        const data = await response.json();

        // Hide typing indicator
        hideTyping();

        // Play receive sound (optional)
        playSound('receive');

        // Add bot message with typing effect
        addMessage('bot', data.answer, formatTimestamp(data.timestamp), data.sources);

        // Speak the response if TTS is enabled
        speakText(data.answer);

    } catch (error) {
        hideTyping();
        addMessage('bot', 'Sorry, an error occurred while processing your question. Please try again.', timestamp, []);
        console.error('Query error:', error);
    }
}

// Suggested prompt click handler with animation
function useSuggestion(text) {
    const input = document.getElementById('queryInput');

    // Animate the button
    event.currentTarget.style.transform = 'scale(0.95)';
    setTimeout(() => {
        event.currentTarget.style.transform = '';
    }, 150);

    // Type effect
    input.value = '';
    typeText(input, text, () => {
        input.focus();
        sendQuery();
    });
}

function typeText(element, text, callback) {
    let i = 0;
    const speed = 30;

    function type() {
        if (i < text.length) {
            element.value += text.charAt(i);
            i++;
            setTimeout(type, speed);
        } else if (callback) {
            setTimeout(callback, 200);
        }
    }

    type();
}

async function loadHistory() {
    try {
        const response = await fetch('/history');
        if (!response.ok) return;

        const history = await response.json();

        if (history.length > 0) {
            hideWelcome();

            history.forEach((item, index) => {
                setTimeout(() => {
                    addMessage('user', item.question, formatTimestamp(item.timestamp));
                    addMessage('bot', item.answer, formatTimestamp(item.timestamp), item.sources);
                }, index * 100);
            });

            setTimeout(scrollToBottom, history.length * 100 + 200);
        }
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// ============================================================
// MESSAGE RENDERING
// ============================================================

function addMessage(type, content, timestamp, sources = []) {
    const container = document.getElementById('chatContainer');

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const iconClass = type === 'user' ? 'fa-user' : 'fa-robot';

    let sourcesHTML = '';
    if (sources && sources.length > 0) {
        sourcesHTML = `
            <div class="message-sources">
                ${sources.map(src => `
                    <span class="source-tag">
                        <i class="fas fa-file-alt"></i>
                        ${escapeHtml(src)}
                    </span>
                `).join('')}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas ${iconClass}"></i>
        </div>
        <div class="message-content">
            <div class="message-bubble">${formatContent(content)}</div>
            ${sourcesHTML}
            <div class="message-time">${timestamp}</div>
        </div>
    `;

    container.appendChild(messageDiv);
    scrollToBottom();

    // Add glow effect on new message
    setTimeout(() => {
        messageDiv.style.filter = 'brightness(1.1)';
        setTimeout(() => {
            messageDiv.style.filter = '';
        }, 300);
    }, 100);
}

function hideWelcome() {
    const welcomeMsg = document.getElementById('welcomeMessage');
    if (welcomeMsg && welcomeMsg.style.display !== 'none') {
        welcomeMsg.style.opacity = '0';
        welcomeMsg.style.transform = 'translateY(-30px) scale(0.95)';
        welcomeMsg.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        setTimeout(() => {
            welcomeMsg.style.display = 'none';
        }, 500);
    }
}

// ============================================================
// TYPING INDICATOR
// ============================================================

function showTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) {
        typing.classList.add('active');
        scrollToBottom();
    }
}

function hideTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) {
        typing.classList.remove('active');
    }
}

// Legacy loading functions
function showLoading() { showTyping(); }
function hideLoading() { hideTyping(); }

// ============================================================
// SOUND EFFECTS (Optional - disabled by default)
// ============================================================

const soundEnabled = false; // Set to true to enable sounds

function playSound(type) {
    if (!soundEnabled) return;

    // Create oscillator for simple beep sounds
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        if (type === 'send') {
            oscillator.frequency.value = 800;
            oscillator.type = 'sine';
        } else {
            oscillator.frequency.value = 600;
            oscillator.type = 'sine';
        }

        gainNode.gain.value = 0.1;
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.1);
    } catch (e) {
        // Audio not supported
    }
}

// ============================================================
// FILE UPLOAD
// ============================================================

function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileSelect);
}

function setupDragDrop() {
    const uploadArea = document.getElementById('uploadArea');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    handleFiles(e.target.files);
}

function handleFiles(files) {
    const allowedExtensions = ['.csv', '.pdf', '.docx', '.txt', '.html'];

    Array.from(files).forEach(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (allowedExtensions.includes(ext)) {
            if (!selectedFiles.some(f => f.name === file.name)) {
                selectedFiles.push(file);
            }
        }
    });

    updateFileList();
    updateUploadButton();
}

function updateFileList() {
    const fileList = document.getElementById('fileList');

    if (selectedFiles.length === 0) {
        fileList.innerHTML = '';
        return;
    }

    const fileIcons = {
        csv: 'fa-file-csv',
        pdf: 'fa-file-pdf',
        docx: 'fa-file-word',
        txt: 'fa-file-lines',
        html: 'fa-file-code'
    };

    fileList.innerHTML = selectedFiles.map((file, index) => {
        const ext = file.name.split('.').pop().toLowerCase();
        const icon = fileIcons[ext] || 'fa-file';
        return `
            <div class="file-item" style="animation-delay: ${index * 0.05}s">
                <i class="fas ${icon}"></i>
                <span>${escapeHtml(file.name)}</span>
                <i class="fas fa-times remove-file" onclick="removeFile(${index})"></i>
            </div>
        `;
    }).join('');
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    updateUploadButton();
}

function updateUploadButton() {
    const uploadBtn = document.getElementById('uploadBtn');
    uploadBtn.disabled = selectedFiles.length === 0;
}

async function uploadFiles() {
    if (selectedFiles.length === 0) return;

    const uploadBtn = document.getElementById('uploadBtn');
    const uploadStatus = document.getElementById('uploadStatus');

    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Processing...</span>';
    uploadStatus.className = 'upload-status';
    uploadStatus.textContent = '';

    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            uploadStatus.className = 'upload-status success';
            uploadStatus.innerHTML = `<i class="fas fa-check-circle"></i> ${data.message}`;
            selectedFiles = [];
            updateFileList();

            // Celebration effect
            createConfetti();
        } else {
            throw new Error(data.detail || 'Upload failed');
        }

    } catch (error) {
        uploadStatus.className = 'upload-status error';
        uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${error.message}`;
        console.error('Upload error:', error);
    }

    uploadBtn.disabled = selectedFiles.length === 0;
    uploadBtn.innerHTML = '<i class="fas fa-bolt"></i> <span>Process & Index</span>';
}

// ============================================================
// CONFETTI CELEBRATION
// ============================================================

function createConfetti() {
    const colors = ['#8b5cf6', '#6366f1', '#22d3ee', '#f472b6', '#34d399', '#fb923c'];
    const confettiCount = 50;

    for (let i = 0; i < confettiCount; i++) {
        const confetti = document.createElement('div');
        confetti.style.cssText = `
            position: fixed;
            width: 10px;
            height: 10px;
            background: ${colors[Math.floor(Math.random() * colors.length)]};
            left: ${Math.random() * 100}vw;
            top: -10px;
            border-radius: ${Math.random() > 0.5 ? '50%' : '0'};
            pointer-events: none;
            z-index: 10000;
            animation: confettiFall ${Math.random() * 2 + 2}s linear forwards;
        `;
        document.body.appendChild(confetti);
        setTimeout(() => confetti.remove(), 4000);
    }
}

// Add confetti keyframes
const confettiStyle = document.createElement('style');
confettiStyle.textContent = `
    @keyframes confettiFall {
        to {
            transform: translateY(100vh) rotate(720deg);
            opacity: 0;
        }
    }
`;
document.head.appendChild(confettiStyle);

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function scrollToBottom() {
    const container = document.getElementById('chatContainer');
    container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
    });
}

function formatTimestamp(timestamp) {
    if (!timestamp) return '';

    try {
        const date = new Date(timestamp.replace(' ', 'T'));
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return timestamp;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatContent(text) {
    let formatted = escapeHtml(text);

    // **bold**
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // *italic*
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // `code`
    formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');

    // Line breaks
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

// ============================================================
// SPEECH-TO-TEXT (Voice Input)
// ============================================================

let recognition = null;
let isRecording = false;

function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        console.warn('Speech Recognition not supported in this browser.');
        const micBtn = document.getElementById('micBtn');
        if (micBtn) {
            micBtn.style.opacity = '0.4';
            micBtn.style.cursor = 'not-allowed';
            micBtn.title = 'Voice input not supported in this browser. Use Chrome or Edge.';
        }
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        isRecording = true;
        const micBtn = document.getElementById('micBtn');
        micBtn.classList.add('recording');
        micBtn.querySelector('i').className = 'fas fa-stop';
        document.getElementById('voiceStatus').textContent = '🎙️ Listening...';
    };

    recognition.onresult = (event) => {
        const input = document.getElementById('queryInput');
        let finalTranscript = '';
        let interimTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        if (finalTranscript) {
            input.value = finalTranscript;
            document.getElementById('voiceStatus').textContent = '✅ Got it!';
        } else {
            input.value = interimTranscript;
            document.getElementById('voiceStatus').textContent = '🎙️ Listening...';
        }
    };

    recognition.onend = () => {
        isRecording = false;
        const micBtn = document.getElementById('micBtn');
        micBtn.classList.remove('recording');
        micBtn.querySelector('i').className = 'fas fa-microphone';

        const input = document.getElementById('queryInput');
        if (input.value.trim()) {
            document.getElementById('voiceStatus').textContent = '';
            sendQuery();
        } else {
            document.getElementById('voiceStatus').textContent = '';
        }
    };

    recognition.onerror = (event) => {
        isRecording = false;
        const micBtn = document.getElementById('micBtn');
        micBtn.classList.remove('recording');
        micBtn.querySelector('i').className = 'fas fa-microphone';

        if (event.error === 'not-allowed') {
            document.getElementById('voiceStatus').textContent = '🚫 Mic permission denied';
            showNotification('Microphone access denied. Please allow mic permission.', 'error');
        } else if (event.error !== 'aborted') {
            document.getElementById('voiceStatus').textContent = '';
        }
    };
}

function toggleMic() {
    if (!recognition) {
        showNotification('Voice input not supported in this browser. Use Chrome or Edge.', 'error');
        return;
    }

    if (isRecording) {
        recognition.stop();
    } else {
        try {
            recognition.start();
        } catch (e) {
            // Already started
            recognition.stop();
        }
    }
}

// ============================================================
// TEXT-TO-SPEECH (Voice Output)
// ============================================================

let ttsEnabled = false;

function toggleTTS() {
    ttsEnabled = !ttsEnabled;
    const btn = document.getElementById('ttsToggle');

    if (ttsEnabled) {
        btn.classList.add('active');
        btn.querySelector('i').className = 'fas fa-volume-high';
        btn.querySelector('span').textContent = 'Voice On';
        showNotification('Voice output enabled', 'success');
    } else {
        btn.classList.remove('active');
        btn.querySelector('i').className = 'fas fa-volume-xmark';
        btn.querySelector('span').textContent = 'Voice Off';
        speechSynthesis.cancel();
        showNotification('Voice output disabled', 'info');
    }
}

function speakText(text) {
    if (!ttsEnabled || !window.speechSynthesis) return;

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    // Clean text for speaking
    const cleanText = text
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/`(.*?)`/g, '$1')
        .replace(/\[Source:.*?\]/g, '')
        .replace(/---/g, '')
        .trim();

    if (!cleanText) return;

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 0.9;

    // Try to use a nice English voice
    const voices = speechSynthesis.getVoices();
    const preferred = voices.find(v =>
        v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Daniel'))
    ) || voices.find(v => v.lang.startsWith('en'));

    if (preferred) {
        utterance.voice = preferred;
    }

    speechSynthesis.speak(utterance);
}

// Preload voices
if (window.speechSynthesis) {
    speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
}

// ============================================================
// CONFIG MODAL (Agent Settings)
// ============================================================

async function openSettings() {
    const modal = document.getElementById('configModal');
    modal.classList.add('active');

    // Load current config from server
    try {
        const response = await fetch('/config');
        if (response.ok) {
            const config = await response.json();
            document.getElementById('cfgModel').value = config.model_name || '';
            document.getElementById('cfgTemp').value = config.temperature || 0;
            document.getElementById('tempValue').textContent = (config.temperature || 0).toFixed(1);
            document.getElementById('cfgTopK').value = config.top_k || 4;
            document.getElementById('cfgChunk').value = config.chunk_size || 1000;
            document.getElementById('cfgPrompt').value = config.system_prompt || '';
        }
    } catch (e) {
        console.error('Failed to load config:', e);
    }

    // Temperature slider live update
    const tempSlider = document.getElementById('cfgTemp');
    tempSlider.oninput = () => {
        document.getElementById('tempValue').textContent = parseFloat(tempSlider.value).toFixed(1);
    };
}

function closeSettings() {
    const modal = document.getElementById('configModal');
    modal.classList.remove('active');
}

async function saveSettings() {
    const config = {
        model_name: document.getElementById('cfgModel').value.trim(),
        temperature: parseFloat(document.getElementById('cfgTemp').value),
        top_k: parseInt(document.getElementById('cfgTopK').value),
        chunk_size: parseInt(document.getElementById('cfgChunk').value),
        system_prompt: document.getElementById('cfgPrompt').value.trim() || null
    };

    try {
        const response = await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (response.ok) {
            showNotification('Agent configuration saved!', 'success');
            closeSettings();
        } else {
            throw new Error('Failed to save config');
        }
    } catch (e) {
        showNotification('Error saving configuration: ' + e.message, 'error');
        console.error('Config save error:', e);
    }
}

// Close modal on overlay click
document.addEventListener('click', (e) => {
    if (e.target.id === 'configModal') {
        closeSettings();
    }
});

// Close modal on Escape
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeSettings();
    }
});

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

document.addEventListener('keydown', (e) => {
    // Ctrl+M = Toggle Mic
    if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
        e.preventDefault();
        toggleMic();
    }
});

// ============================================================
// INIT VOICE ON PAGE LOAD
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
});

