import { useState, useRef, useEffect } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', content: 'Greetings adventurer! Ask me anything about D&D, and I will consult the sacred texts.', sources: [] }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    
    const userMessage = { role: 'user', content: input, sources: [] }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      })
      
      if (!response.ok) throw new Error('API request failed')
      
      const data = await response.json()
      
      // Deduplicate sources by source file name for a cleaner UI
      const uniqueSourcesMap = new Map()
      const rawSources = data.sources || []
      rawSources.forEach(src => {
          if (!uniqueSourcesMap.has(src.source)) {
              uniqueSourcesMap.set(src.source, src)
          }
      })
      const filteredSources = Array.from(uniqueSourcesMap.values())

      setMessages(prev => [...prev, {
        role: 'bot',
        content: data.answer,
        sources: filteredSources
      }])
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'bot',
        content: 'I encountered an error while consulting the database. Make sure the FastAPI backend is running.',
        sources: []
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      <div className="header">
        <h1>
          <div className="header-accent"></div>
          D&D Codex Agent
        </h1>
      </div>
      
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.role}`}>
              <div className="message">
                {msg.content}
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  {msg.sources.map((src, sIdx) => (
                    <div key={sIdx} className="source-badge">
                      <span className="source-title">{src.source}</span>
                      {src.page && <span>Page: {src.page}</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="message-wrapper bot">
              <div className="message">
                <div className="loading-indicator">
                  <div className="dot"></div>
                  <div className="dot"></div>
                  <div className="dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <form className="input-area" onSubmit={handleSend}>
          <input 
            type="text" 
            className="input-box"
            placeholder="Ask about classes, spells, or monster stats..." 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" className="send-btn" disabled={!input.trim() || isLoading}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
      </div>
    </div>
  )
}

export default App
