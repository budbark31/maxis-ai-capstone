import React, { useEffect, useRef, useState } from "react";
import "./App.css";

// Icons
const SendIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 12h14m-7-7l7 7-7 7" />
  </svg>
);
const SourceIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ marginLeft: 6, verticalAlign: "-2px" }}>
    <path strokeWidth="2" d="M7 17L17 7M10 7h7v7" />
  </svg>
);

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
    const user = { sender: "user", text: text.trim(), sources: [] };
    setMessages((m) => [...m, user]);
    setInput("");
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: user.text })
      });
      const data = await res.json();
      const bot = {
        sender: "bot",
        text: data?.reply || data?.answer || "I couldn't generate a response.",
        sources: Array.isArray(data?.sources) ? data.sources : []
      };
      setMessages((m) => [...m, bot]);
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
                <SendIcon />
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
                  <p className="bubble-text">{msg.text}</p>
                  {!isUser && msg.sources?.length > 0 && (
                    <div className="sources">
                      <div className="sources-title">Sources</div>
                      <ul>
                        {msg.sources.map((u, i) => (
                          <li key={i}>
                            <a href={u} target="_blank" rel="noreferrer" className="source-link">
                              {u.length > 72 ? `${u.slice(0, 72)}…` : u}
                              <SourceIcon />
                            </a>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              );
            })}
            {isLoading && (
              <div className="chat-bubble bot">
                <div className="typing-dots">
                  <span></span><span></span><span></span>
                </div>
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
                <SendIcon />
              </button>
            </div>
          </form>
        </main>
      )}
    </div>
  );
}

