import { useState, useEffect } from 'react';

export default function Home() {
  // Hard-coded sessionId for simplicity
  const [sessionId, setSessionId] = useState('my-session-id');
  
  // The user’s message and the AI’s latest response
  const [userInput, setUserInput] = useState('');
  const [latestResponse, setLatestResponse] = useState('');

  // Entire chat history from the backend
  const [chatHistory, setChatHistory] = useState([]);

  // Fetch the chat history on initial load or session ID change
  useEffect(() => {
    getHistory();
  }, [sessionId]);

  // Helper to call your /history endpoint
  async function getHistory() {
    try {
      const response = await fetch('http://localhost:8000/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setChatHistory(prev => [...prev, `History: ${data.response}`]);
    } catch (err) {
      console.error('Failed to fetch history:', err);
    }
  }

  // Helper to call your /chat endpoint
  async function sendMessage() {
    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: userInput,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      setLatestResponse(data.response);
      setChatHistory(prev => [...prev, `User: ${userInput}`, `AI: ${data.response}`]);

      // Clear the input field
      setUserInput('');
    } catch (err) {
      console.error('Failed to send message:', err);
    }
  }

  return (
    <div style={{ maxWidth: '600px', margin: 'auto', padding: '2rem' }}>
      <h1>My AI Chat</h1>

      <div style={{ marginBottom: '1rem' }}>
        <label htmlFor="sessionId">Session ID:</label>
        <input
          id="sessionId"
          type="text"
          value={sessionId}
          onChange={(e) => setSessionId(e.target.value)}
          style={{ marginLeft: '0.5rem' }}
        />
        <button onClick={getHistory} style={{ marginLeft: '1rem' }}>
          Refresh History
        </button>
      </div>

      <div style={{ marginBottom: '1rem' }}>
        <label htmlFor="userInput">Your Message:</label><br />
        <textarea
          id="userInput"
          rows={3}
          style={{ width: '100%' }}
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
        />
        <button onClick={sendMessage} style={{ marginTop: '0.5rem' }}>
          Send
        </button>
      </div>

      <div style={{ marginBottom: '1rem', background: '#f1f1f1', padding: '1rem' }}>
        <h2>Latest Response</h2>
        <p>{latestResponse || "No response yet."}</p>
      </div>

      <div style={{ background: '#fafafa', padding: '1rem' }}>
        <h2>Chat History</h2>
        <ul>
          {chatHistory.map((text, index) => (
            <li key={index}>{text}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}