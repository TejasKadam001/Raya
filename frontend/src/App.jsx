import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Plus, Send, Paperclip, Globe, Search, Command, ArrowUpRight } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const handleInput = (e) => {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 150)}px`;
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: input,
      });
      setMessages(prev => [...prev, { role: 'bot', content: response.data.response }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { role: 'bot', content: 'Sync error. Reconnect to Raya cluster.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const quickPrompts = [
    "Synthesize the future of AGI",
    "Develop a neural network in PyTorch",
    "Optimize my React app's performance"
  ];

  return (
    <>
      <div className="sidebar">
        <div className="new-chat-btn" onClick={() => setMessages([])}>
          <Plus size={18} />
          <span>New Intel</span>
        </div>
        <div style={{ flex: 1 }}></div>
        <div className="new-chat-btn" style={{ border: 'none', opacity: 0.5 }}>
          <Command size={16} />
          <span style={{ fontSize: '0.8rem' }}>Node v1.0.4</span>
        </div>
      </div>

      <div className="main-content">
        {messages.length === 0 ? (
          <div className="hero">
            <h1 className="raya-logo">raya</h1>
            <p className="hero-subtitle">A garage-built generative brain.</p>
            <div className="features-grid">
              {quickPrompts.map((prompt, i) => (
                <div key={i} className="feature-card" onClick={() => setInput(prompt)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <ArrowUpRight size={14} />
                  </div>
                  <p>{prompt}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-container" ref={chatContainerRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role === 'user' ? 'user-row' : 'bot-row'}`}>
                <div className={`message-bubble ${msg.role === 'user' ? 'user-bubble' : 'bot-bubble'}`}>
                  <span className="sender-label">{msg.role === 'user' ? 'User' : 'Raya'}</span>
                  <div style={{ color: msg.role === 'user' ? '#fff' : '#ececec' }}>
                    {msg.content}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message-row bot-row">
                <div className="message-bubble bot-bubble" style={{ opacity: 0.6 }}>
                  <span className="sender-label">Raya</span>
                  <div className="typing-loader">
                    <span>.</span><span>.</span><span>.</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="input-container">
          <div className="input-wrapper">
            <button className="icon-btn">
              <Paperclip size={20} />
            </button>
            <textarea
              ref={textareaRef}
              className="chat-input"
              placeholder="Query Raya..."
              rows={1}
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
            />
            <div className="actions-row">
              <button className="icon-btn">
                <Globe size={20} />
              </button>
              <button className="icon-btn">
                <Search size={20} />
              </button>
              <button
                className="send-btn"
                onClick={handleSend}
                disabled={!input || isLoading}
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
