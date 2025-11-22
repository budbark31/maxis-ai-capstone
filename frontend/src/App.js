import React, { useEffect, useRef, useState } from "react";
import ReactMarkdown from 'react-markdown'; 
import "./App.css";

// Helper to clean URLs for display (e.g. "https://marywood.edu/..." -> "marywood.edu/...")
const getSourceLabel = (url) => {
  try {
    const urlObj = new URL(url);
    let label = urlObj.hostname.replace('www.', '') + urlObj.pathname;
    // Truncate if too long for the box
    if (label.length > 25) return label.substring(0, 25) + '...';
    return label;
  } catch (e) {
    return "Source Link";
  }
};

const API_BASE = "http://localhost:8000";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef(null);

  const suggestions = [
    "What’s happening on campus this week?",
    "How do I access the Marywood VPN?",
    "How can I contact the Registrar’s Office?",
    "Tell me about athletics at Marywood."
  ];

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(text) {
    if (!text?.trim()) return;
    
    // 1. Update UI immediately with User Message
    const newUserMsg = { sender: "user", text: text.trim(), sources: [] };
    const newHistory = [...messages, newUserMsg];
    setMessages(newHistory);
    setInput("");
    setIsLoading(true);

    try {
      // 2. Send Message + History to Backend
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: newUserMsg.text,
          // Send last 10 messages for context
          history: newHistory.slice(-10) 
        })
      });
      
      const data = await res.json();
      
      const botMsg = {
        sender: "bot",
        text: data?.reply || "I couldn't generate a response.",
        sources: Array.isArray(data?.sources) ? data.sources : []
      };
      setMessages((m) => [...m, botMsg]);
    } catch (e) {
      setMessages((m) => [...m, { sender: "bot", text: `Network error. Is the backend running at ${API_BASE}?`, sources: [] }]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(e) {
    e.preventDefault();
    sendMessage(input);
  }

  return (
    <div className="app">
      {messages.length === 0 ? (
        <main className="container welcome animate-fade-in">
          <div className="logo-wrap">
            <img
              src={process.env.PUBLIC_URL + "/marywood-university-athletics-logo.png"}
              alt="Marywood University Athletics logo"
              className="logo-hero"
            />
          </div>
          <h1 className="title">
            Hello, I’m <span className="gold">Maxis.ai</span>
          </h1>
          <p className="subtitle">Ask me anything about Marywood University.</p>

          <div className="welcome-grid">
            {suggestions.map((s, i) => (
              <button key={i} className="suggested-card" onClick={() => sendMessage(s)}>
                {s}
              </button>
            ))}
          </div>

          <form className="prompt-row" onSubmit={handleSubmit}>
            <div className="prompt-shell">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask Maxis…"
                aria-label="Your message"
              />
              <button type="submit" disabled={!input.trim() || isLoading} title="Send">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7l7 7-7 7" /></svg>
              </button>
            </div>
          </form>
        </main>
      ) : (
        <main className="container chat">
          <div className="chat-scroller">
            {messages.map((msg, idx) => {
              const isUser = msg.sender === "user";
              return (
                <div key={idx} className={`chat-bubble ${isUser ? "user" : "bot"} animate-fade-in`}>
                  
                  {/* Render Markdown Text */}
                  <div className="bubble-text prose">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                  
                  {/* Source Boxes UI */}
                  {!isUser && msg.sources?.length > 0 && (
                    <div className="sources-container">
                      <div className="sources-header" style={{fontSize: '0.75rem', fontWeight: 'bold', color: '#6b7280', marginTop: '12px', marginBottom: '4px'}}>Sources</div>
                      <div className="sources-grid">
                        {msg.sources.map((u, i) => (
                          <a key={i} href={u} target="_blank" rel="noreferrer" className="source-card">
                            {/* Generic Globe Icon */}
                            <svg className="source-favicon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
                            <span className="source-title">{getSourceLabel(u)}</span>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
            {isLoading && (
              <div className="chat-bubble bot">
                <div className="typing-dots"><span></span><span></span><span></span></div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          <form className="prompt-row" onSubmit={handleSubmit}>
            <div className="prompt-shell">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask Maxis…"
                aria-label="Your message"
              />
              <button type="submit" disabled={!input.trim() || isLoading} title="Send">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7l7 7-7 7" /></svg>
              </button>
            </div>
          </form>
        </main>
      )}
    </div>
  );
}