
import React, { useState } from 'react';
import './MessageHistory.css';

function MessageHistory({ messages, isOpen, onClose, onClearHistory, onSpeak, activeMessageId }) {
    const [showConfirmDialog, setShowConfirmDialog] = useState(false);

    const formatMessage = (text) => {
        const lines = text.split('\n').map(l => l.trim()).filter(l => l !== '');

        let parts = { opening: [], shloka: [], explanation: [], steps: [] };
        let currentState = 'opening';

        lines.forEach((line) => {
            // Transition to Shloka
            if (currentState === 'opening' && (line.includes('भगवद गीता') || line.includes('Bhagavad Gita') || line.includes('シュローカ') || line.includes('ギーター') || line.includes('Chapter'))) {
                currentState = 'shloka';
                parts.shloka.push(line);
                return;
            }

            // Transition to Explanation
            if (currentState === 'shloka') {
                const isShlokaLine = line.match(/[।॥|‖]/) || line.includes('Chapter') || line.includes('भगवद गीता') || line.includes('Bhagavad Gita') || line.includes('シュローカ') || line.includes('ギーター');
                if (!isShlokaLine && line.length > 0) {
                    // Try to guess if it's still Sanskrit based on Devanagari chars without dandas
                    const hasDevanagari = /[\u0900-\u097F]/.test(line);
                    if (!hasDevanagari) {
                        currentState = 'explanation';
                        parts.explanation.push(line);
                        return;
                    }
                }
            }

            // Transition to Steps
            if (currentState === 'explanation') {
                if (line.endsWith('：') || line.endsWith(':') || /^\d+\./.test(line) || /^[１-９]．/.test(line)) {
                    currentState = 'steps';
                    parts.steps.push(line);
                    return;
                }
            }

            if (parts[currentState]) {
                parts[currentState].push(line);
            }
        });

        // Fallback if parsing failed completely or it's a short text (e.g. user messages)
        if (parts.shloka.length === 0 && parts.explanation.length === 0 && parts.steps.length === 0) {
            return <div className="message-box general-box">{lines.map((l, i) => <p key={i}>{l}</p>)}</div>;
        }

        return (
            <div className="four-part-message">
                {parts.opening.length > 0 && (
                    <div className="message-box opening-box">
                        {parts.opening.map((l, i) => <p key={i}>{l}</p>)}
                    </div>
                )}
                {parts.shloka.length > 0 && (
                    <div className="message-box shloka-box">
                        {parts.shloka.map((l, i) => <div key={i} className="shloka-line">{l}</div>)}
                    </div>
                )}
                {parts.explanation.length > 0 && (
                    <div className="message-box explanation-box">
                        {parts.explanation.map((l, i) => <p key={i}>{l}</p>)}
                    </div>
                )}
                {parts.steps.length > 0 && (
                    <div className="message-box steps-box">
                        {parts.steps.map((l, i) => <p key={i}>{l}</p>)}
                    </div>
                )}
            </div>
        );
    };

    const handleClearHistory = () => {
        onClearHistory();
        setShowConfirmDialog(false);
    };

    return (
        <>
            {/* Overlay */}
            {isOpen && <div className="history-overlay" onClick={onClose}></div>}

            {/* Sidebar */}
            <div className={`message-history ${isOpen ? 'open' : ''}`}>
                <div className="history-header">
                    <h2>Divine Dialogue</h2>
                    <div className="header-actions">
                        {messages.length > 0 && (
                            <button
                                className="clear-history-button"
                                onClick={() => setShowConfirmDialog(true)}
                                title="Clear history"
                            >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            </button>
                        )}
                        <button className="close-button" onClick={onClose}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path d="M18 6L6 18M6 6l12 12" strokeWidth="2" strokeLinecap="round" />
                            </svg>
                        </button>
                    </div>
                </div>

                <div className="history-content">
                    {messages.length === 0 ? (
                        <p className="empty-state">Your journey has just begun...</p>
                    ) : (
                        messages.map((message) => (
                            <div key={message.id} className={`history-message ${message.type}`}>
                                <div className="message-header">
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                                        <span className="message-author">
                                            {message.type === 'krishna' ? '🪈 Krishna' : '👤 You'}
                                        </span>
                                        {message.type === 'krishna' && (
                                            <button
                                                className={`speak-button ${activeMessageId === message.id ? 'speaking' : ''}`}
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onSpeak(message.text, message.id);
                                                }}
                                                title={activeMessageId === message.id ? "Stop reading" : "Read aloud"}
                                            >
                                                {activeMessageId === message.id ? (
                                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                        <rect x="6" y="6" width="12" height="12" />
                                                    </svg>
                                                ) : (
                                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                        <path d="M11 5L6 9H2V15H6L11 19V5Z" fill="currentColor" />
                                                        <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                                                        <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
                                                    </svg>
                                                )}
                                            </button>
                                        )}
                                    </div>
                                    <span className="message-time">
                                        {(() => {
                                            try {
                                                if (!message.timestamp) return '';
                                                let date;

                                                if (message.timestamp instanceof Date) {
                                                    date = message.timestamp;
                                                } else {
                                                    // Robust handling for SQLite strings and Pre-formatted ISO strings
                                                    let ts = String(message.timestamp);

                                                    // Convert SQLite spacing "2024-10-10 12:00:00" -> "2024-10-10T12:00:00"
                                                    if (ts.includes(' ') && !ts.includes('T')) {
                                                        ts = ts.replace(' ', 'T');
                                                    }

                                                    // Add 'Z' (UTC) only if timezone modifier is entirely missing
                                                    if (!ts.endsWith('Z') && !ts.includes('+') && ts.split('-').length <= 3) {
                                                        ts += 'Z';
                                                    }

                                                    date = new Date(ts);
                                                }

                                                // Absolute fallback if parsing failed entirely
                                                if (isNaN(date.getTime())) {
                                                    date = new Date(message.timestamp);
                                                }

                                                // Rather than printing "Invalid Date", quietly return blank if completely broken
                                                if (isNaN(date.getTime())) return '';

                                                return date.toLocaleTimeString('en-IN', {
                                                    timeZone: 'Asia/Kolkata',
                                                    hour: '2-digit',
                                                    minute: '2-digit'
                                                });
                                            } catch (e) {
                                                return '';
                                            }
                                        })()}
                                    </span>
                                </div>
                                <div className="message-body">
                                    {message.type === 'krishna' ? (
                                        formatMessage(message.text)
                                    ) : (
                                        <p>{message.text}</p>
                                    )}
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Confirmation Dialog */}
            {showConfirmDialog && (
                <div className="confirm-overlay" onClick={() => setShowConfirmDialog(false)}>
                    <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
                        <div className="confirm-icon">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                        </div>
                        <h3>Clear Chat History?</h3>
                        <p>This will permanently delete all your conversations. This action cannot be undone.</p>
                        <div className="confirm-actions">
                            <button className="btn-cancel" onClick={() => setShowConfirmDialog(false)}>
                                Cancel
                            </button>
                            <button className="btn-confirm" onClick={handleClearHistory}>
                                Clear History
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}

export default React.memo(MessageHistory);
