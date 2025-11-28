// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentPdfText = '';
let uploadedFilename = '';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const numQuestions = document.getElementById('numQuestions');
const uploadStatus = document.getElementById('uploadStatus');
const summaryText = document.getElementById('summaryText');
const summarizeBtn = document.getElementById('summarizeBtn');
const qaList = document.getElementById('qaList');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const statusText = document.getElementById('statusText');
const docCount = document.getElementById('docCount');
const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    setInterval(checkHealth, 30000); // Check health every 30 seconds
});

// Setup Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--primary-color)';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = 'var(--border-color)';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = 'var(--border-color)';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect();
        }
    });
    
    // Upload button
    uploadBtn.addEventListener('click', uploadPDF);
    
    // Summarize button
    summarizeBtn.addEventListener('click', generateSummary);
    
    // Chat send button
    sendBtn.addEventListener('click', sendMessage);
    
    // Chat input enter key
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}

// Check API Health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        statusText.textContent = '✅ Online';
        statusText.style.color = 'var(--success-color)';
        docCount.textContent = `${data.document_count} docs`;
        
        // Enable chat if documents exist
        if (data.document_count > 0) {
            chatInput.disabled = false;
            sendBtn.disabled = false;
        }
    } catch (error) {
        statusText.textContent = '❌ Offline';
        statusText.style.color = 'var(--danger-color)';
        console.error('Health check failed:', error);
    }
}

// Handle File Selection
function handleFileSelect() {
    const file = fileInput.files[0];
    if (file) {
        uploadArea.querySelector('p').textContent = `Selected: ${file.name}`;
        uploadBtn.disabled = false;
    }
}

// Upload PDF with Q&A Generation
async function uploadPDF() {
    const file = fileInput.files[0];
    if (!file) {
        showStatus('Please select a PDF file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    const numQ = numQuestions.value;
    
    showLoading(`Uploading and generating ${numQ} Q&A pairs...`);
    
    try {
        const response = await fetch(`${API_BASE_URL}/upload-with-qa?num_questions=${numQ}`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            uploadedFilename = data.filename;
            // Store the full text if available
            currentPdfText = data.full_text || '';
            
            console.log('Upload successful:', {
                filename: data.filename,
                textLength: currentPdfText.length,
                qaGenerated: data.total_qa_generated
            });
            
            showStatus(`✅ Success! Generated ${data.total_qa_generated} Q&A pairs`, 'success');
            
            // Display Q&A pairs
            displayQAPairs(data.qa_pairs);
            
            // Enable summary button (always enable it now, will fallback to Gemini if needed)
            summarizeBtn.disabled = false;
            console.log('Summary button enabled. Has text:', !!currentPdfText);
            
            // Enable chat
            chatInput.disabled = false;
            sendBtn.disabled = false;
            
            // Update health
            checkHealth();
            
            // Add success message to chat
            addMessage('bot', `✅ Successfully uploaded ${data.filename}! You can now ask questions about it.`);
        } else {
            showStatus(`❌ Error: ${data.detail}`, 'error');
        }
    } catch (error) {
        showStatus(`❌ Upload failed: ${error.message}`, 'error');
        console.error('Upload error:', error);
    } finally {
        hideLoading();
    }
}

// Display Q&A Pairs
function displayQAPairs(qaPairs) {
    if (!qaPairs || qaPairs.length === 0) {
        qaList.innerHTML = '<p class="placeholder">No Q&A pairs generated</p>';
        return;
    }
    
    qaList.innerHTML = qaPairs.map((qa, index) => `
        <div class="qa-item">
            <div class="qa-question">Q${index + 1}: ${qa.question}</div>
            <div class="qa-answer">A: ${qa.answer}</div>
        </div>
    `).join('');
}

// Generate Summary
async function generateSummary() {
    if (!uploadedFilename) {
        showStatus('Please upload a PDF first', 'error');
        return;
    }
    
    console.log('Generating summary. Text available:', currentPdfText ? currentPdfText.length : 0);
    
    if (!currentPdfText) {
        showStatus('⚠️ Document text not found. Using chat-based summary instead...', 'info');
        // Fallback: use chat to get summary
        try {
            showLoading('Generating summary with Gemini...');
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: "Please provide a comprehensive summary of this document, covering the main topics, key concepts, and important takeaways.",
                    top_k: 10
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                summaryText.innerHTML = `
                    <p><strong>Document:</strong> ${uploadedFilename}</p>
                    <p style="margin-top: 1rem;">${data.answer}</p>
                `;
                showStatus('✅ Summary generated with AI', 'success');
            } else {
                throw new Error(data.detail || 'Failed to generate summary');
            }
        } catch (error) {
            summaryText.innerHTML = `
                <p><strong>Document:</strong> ${uploadedFilename}</p>
                <p class="placeholder">Failed to generate summary: ${error.message}</p>
            `;
            showStatus(`❌ Summary failed: ${error.message}`, 'error');
        } finally {
            hideLoading();
        }
        return;
    }
    
    showLoading('Generating summary with DistilBART...');
    
    try {
        // Call the /summarize endpoint with the stored PDF text
        const response = await fetch(`${API_BASE_URL}/summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: currentPdfText,
                max_length: 250,
                min_length: 80
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            summaryText.innerHTML = `
                <p><strong>Document:</strong> ${uploadedFilename}</p>
                <p style="margin-top: 1rem;">${data.summary}</p>
                <p style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-secondary);">
                    Original: ${data.original_length.toLocaleString()} chars | 
                    Summary: ${data.summary_length} chars
                </p>
            `;
            showStatus('✅ Summary generated successfully', 'success');
        } else {
            throw new Error(data.detail || 'Summarization failed');
        }
    } catch (error) {
        summaryText.innerHTML = `
            <p><strong>Document:</strong> ${uploadedFilename}</p>
            <p class="placeholder">Failed to generate summary: ${error.message}</p>
        `;
        showStatus(`❌ Summary failed: ${error.message}`, 'error');
        console.error('Summary error:', error);
    } finally {
        hideLoading();
    }
}

// Send Chat Message
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage('user', message);
    chatInput.value = '';
    
    // Add typing indicator
    const typingId = addTypingIndicator();
    
    try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: message,
                top_k: 5
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator(typingId);
        
        if (response.ok) {
            // Add bot response
            const contextInfo = data.context_used 
                ? `📚 (Used ${data.relevant_chunks} document chunks)`
                : '⚠️ (No relevant context found)';
            
            addMessage('bot', `${data.answer}\n\n<small style="color: var(--text-secondary);">${contextInfo}</small>`);
        } else {
            addMessage('bot', `❌ Error: ${data.detail}`);
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage('bot', `❌ Error: ${error.message}`);
        console.error('Chat error:', error);
    }
}

// Add Message to Chat
function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${content}</p>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add Typing Indicator
function addTypingIndicator() {
    const id = 'typing-' + Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.id = id;
    messageDiv.className = 'message bot loading';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

// Remove Typing Indicator
function removeTypingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// Show Status Message
function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
    
    setTimeout(() => {
        uploadStatus.className = 'status-message';
    }, 5000);
}

// Show Loading Overlay
function showLoading(message) {
    loadingText.textContent = message;
    loadingOverlay.classList.add('active');
}

// Hide Loading Overlay
function hideLoading() {
    loadingOverlay.classList.remove('active');
}
