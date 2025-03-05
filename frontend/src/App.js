import React, { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [showReadReceipt, setShowReadReceipt] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    // Add some initial messages (bot + system)
    setMessages([
      { id: 1, sender: 'system', text: 'Today 7:16 pm' }, // "system" for center text or top bar
      { id: 2, sender: 'bot', text: 'Hello! How can I help you?', timestamp: '7:16 pm' },
    ]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Grab current time (HH:MM)
    const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Create user message
    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: input.trim(),
      timestamp: timeNow,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setShowReadReceipt(false);

    // Simulate "typing" indicator for 1.5s before bot replies
    setTyping(true);
    setTimeout(() => {
      setTyping(false);
      setShowReadReceipt(true);
      const botReply = {
        id: Date.now() + 1,
        sender: 'bot',
        text: 'This is a bot reply!',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, botReply]);
    }, 1500);
  };

  return (
    <div className="app-container">
      <div className="chat-container">
        {/* You can also place a time header here if you want it top-right instead of system center */}
        {/* <div className="header-time">Today 7:16 pm</div> */}

        <div className="messages-container">
          {messages.map((msg) => {
            if (msg.sender === 'system') {
              // Centered "system" text (e.g. "Today 7:16 pm")
              return (
                <div
                  key={msg.id}
                  style={{ textAlign: 'center', margin: '8px 0', color: '#8e8e93' }}
                >
                  {msg.text}
                </div>
              );
            }

            // Normal user/bot message
            return (
              <div key={msg.id} className={`message-wrapper ${msg.sender}-message`}>
                <div className="message-content">
                  <div>{msg.text}</div>
                  {msg.timestamp && (
                    <span className="message-time">{msg.timestamp}</span>
                  )}
                  {/* "Read" receipt for user messages after bot responds */}
                  {msg.sender === 'user' && showReadReceipt && (
                    <span className="read-receipt">Read {msg.timestamp}</span>
                  )}
                </div>
              </div>
            );
          })}

          {/* Typing indicator bubble */}
          {typing && (
            <div className="message-wrapper bot-message">
              <div className="typing-indicator-bubble">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Smaller, centered input bar */}
        <div className="input-container">
          <form onSubmit={handleSend}>
            <input
              type="text"
              placeholder="Ask anything"
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button className="send-button" type="submit">
              ➤
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
