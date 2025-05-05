"use client";  // enable client-side features like hooks or local component state

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { v4 as uuidv4 } from 'uuid';

// This is a simple Next.js 13 App Router page that mirrors your previous index.js page functionality.
export default function Home() {
  // Generate a new UUID for the initial session
  const [sessionId, setSessionId] = useState(uuidv4());
  
  // Entire chat history from the backend
  const [chatHistory, setChatHistory] = useState<string[]>([]);

  // Track user input
  const [userInput, setUserInput] = useState("");

  // Sidebar width dedicated to session ID
  const sidebarWidth = "400px";

  // Ref for the scrollable container in the right pane
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  // Ref for the dummy div at the bottom of the chat for scrolling
  const bottomRef = useRef<HTMLDivElement>(null);

  // 1) Add new state for storing available sessions
  type Session = {
    session_id: string;
    last_message: string;
    message_type: string;
    created_at: string;
  };

  const [sessions, setSessions] = useState<Record<string, Session[]>>({});

  // Scroll to the bottom whenever chatHistory changes
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory]);

  // Fetch the chat history on initial load or when sessionId changes
  useEffect(() => {
    getHistory();
  }, [sessionId]);

  // 2) Fetch the sessions list on mount (or any time you choose)
  useEffect(() => {
    fetch("http://localhost:8000/sessions")
      .then((res) => res.json())
      .then((data) => {
        if (data && data.sessions) {
          setSessions(data.sessions);
        }
      })
      .catch((err) => console.error("Failed to fetch sessions:", err));
  }, []);

  // Helper to call your /history endpoint
  async function getHistory() {
    try {
      const response = await fetch("http://localhost:8000/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      const data = await response.json();
      // data.response should be an array of chat messages
      setChatHistory(data.response);
    } catch (err) {
      setChatHistory([]);
    }
  }

  // Updated sendMessage to use streaming from the new endpoint.
  async function sendMessage() {
    if (!userInput.trim()) return;

    // Append the user message.
    setChatHistory((prev) => [...prev, `User: ${userInput}`]);
    const message = userInput;
    setUserInput("");

    // Append an entry for the upcoming AI response.
    setChatHistory((prev) => [...prev, `AI: `]);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: message,
          session_id: sessionId,
        }),
      });

      // Read from the response stream.
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let streamedText = "";

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            const chunk = decoder.decode(value);
            streamedText += chunk;
            // Update the last entry in chatHistory (the AI message) with the streaming text.
            setChatHistory((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = `AI: ${streamedText}`;
              return updated;
            });
          }
        }
      }
    } catch (err) {
      console.error("Error streaming chat:", err);
    }
  }

  // Handle key down event – send message on pressing Enter (ignoring Shift+Enter)
  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  // Helper to remove the prefixes "User:" and "AI:" from messages.
  function stripPrefix(text: string): string {
    return text.replace(/^(User:|AI:)\s*/, "");
  }

  // Add delete session function
  async function deleteSession(sessionId: string, e: React.MouseEvent) {
    e.stopPropagation(); // Prevent triggering the conversation click
    if (!confirm('Are you sure you want to delete this conversation?')) return;
    
    try {
      const response = await fetch("http://localhost:8000/delete-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (response.ok) {
        // Refresh the sessions list
        const sessionsResponse = await fetch("http://localhost:8000/sessions");
        const sessionsData = await sessionsResponse.json();
        if (sessionsData && sessionsData.sessions) {
          setSessions(sessionsData.sessions);
        }
        
        // If the deleted session was the current one, create a new session
        if (sessionId === sessionId) {
          setSessionId(uuidv4());
          setChatHistory([]);
        }
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
    }
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* LEFT SIDEBAR: Dedicated to Session ID */}
      <div
        style={{
          width: sidebarWidth,
          padding: "1rem",
          borderRight: "1px solid #ccc",
          boxSizing: "border-box",
          background: "#fff",
          position: "fixed",
          top: 0,
          left: 0,
          height: "100vh",
          overflowY: "auto",
        }}
      >

        {/* 3) Show the fetched sessions and let the user click to switch */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
          <h3 style={{ fontWeight: "bold", marginBottom: 0 }}>Previous Conversations</h3>
          <button
            onClick={async () => {
              try {
                const res = await fetch("http://localhost:8000/create-session", { method: "POST" });
                const data = await res.json();
                if (data && data.session_id) {
                  setSessionId(data.session_id);
                  setChatHistory([]);
                  // Fetch updated sessions list
                  const sessionsResponse = await fetch("http://localhost:8000/sessions");
                  const sessionsData = await sessionsResponse.json();
                  if (sessionsData && sessionsData.sessions) {
                    setSessions(sessionsData.sessions);
                  }
                }
              } catch (err) {
                console.error("Failed to create new session:", err);
              }
            }}
            style={{
              background: "#f5f5f5",
              border: "none",
              borderRadius: "50%",
              width: "2rem",
              height: "2rem",
              fontSize: "1.5rem",
              fontWeight: "bold",
              color: "#333",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginLeft: "0.5rem"
            }}
            title="New conversation"
          >
            +
          </button>
        </div>
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {Object.entries(sessions).map(([period, periodSessions]) => (
            <div key={period} style={{ marginBottom: "1.5rem" }}>
              <h4 style={{ 
                fontSize: "0.9em", 
                color: "black", 
                marginBottom: "0.5rem",
                paddingLeft: "1.2rem",
                fontWeight: "bold"
              }}>
                {period}
              </h4>
              <ul style={{ paddingInlineStart: "1.2rem" }}>
                {periodSessions.map((session: Session) => (
                  <li
                    key={session.session_id}
                    onClick={() => {
                      setSessionId(session.session_id);
                      getHistory(); // load that session's history
                    }}
                    style={{ 
                      cursor: "pointer", 
                      marginBottom: "0.5rem",
                      padding: "0.5rem",
                      border: "1px solid #eee",
                      borderRadius: "4px",
                      backgroundColor: "#f9f9f9"
                    }}
                  >
                    <div style={{ 
                      fontSize: "0.9em", 
                      color: "#666",
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      maxWidth: "100%",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center"
                    }}>
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
                        {typeof session.last_message === 'string' 
                          ? session.last_message.length > 50 
                            ? session.last_message.substring(0, 50) + "..."
                            : session.last_message
                          : JSON.stringify(session.last_message)}
                      </span>
                      <button
                        onClick={(e) => deleteSession(session.session_id, e)}
                        style={{
                          background: "none",
                          border: "none",
                          color: "#999",
                          cursor: "pointer",
                          padding: "4px",
                          marginLeft: "8px",
                          fontSize: "1.1em",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center"
                        }}
                        title="Delete conversation"
                      >
                        ×
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </ul>
      </div>

      {/* RIGHT MAIN CONTENT AREA */}
      <div
        style={{
          marginLeft: sidebarWidth,
          display: "flex",
          flexDirection: "column",
          height: "100vh",
        }}
      >
        {/* Chat History */}
        <div
          ref={chatContainerRef}
          style={{
            flex: 1,
            width: "100%", // Ensures the chat container spans the full width
            padding: "1rem",
            paddingBottom: "80px", // Extra space to ensure latest messages are visible
            overflowY: "auto",
            boxSizing: "border-box",
            background: "#ffffff",
          }}
        >
          <div style={{ marginBottom: "1rem" }}>
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {chatHistory.map((text, index) => (
                <li key={index} style={{ marginBottom: "0.5rem" }}>
                  <div
                    style={{
                      display: "inline-block",
                      background: text.startsWith("User:") ? "#f0f0f0" : "#ffffff",
                      padding: "0.5rem",
                      borderRadius: "10px",
                      wordBreak: "break-word",
                      marginLeft: text.startsWith("User:") ? "15rem" : "auto",
                      marginRight: text.startsWith("User:") ? "auto" : "1rem",
                      maxWidth: "60%", // Optional: limit the width of messages
                    }}
                  >
                    <ReactMarkdown>{stripPrefix(text)}</ReactMarkdown>
                  </div>
                </li>
              ))}
              {/* Dummy div for scrolling */}
              <div ref={bottomRef} />
            </ul>
          </div>
        </div>

        {/* Fixed Message Input at the Bottom (Prompt) */}
        <div
          style={{
            background: "#ffffff",
            padding: "1rem 2rem",
            boxShadow: "0 -2px 10px rgba(0,0,0,0.1)",
            boxSizing: "border-box",
            position: "fixed",
            bottom: 0,
            left: sidebarWidth,
            width: `calc(100vw - ${sidebarWidth})`,
            display: "flex",
            justifyContent: "center",
          }}
        >
          <input
            type="text"
            placeholder="Type your message..."
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={handleKeyDown}
            style={{
              width: "100%", // Approximately 1/3 of the full screen width
              padding: "0.5rem",
              boxSizing: "border-box",
            }}
          />
        </div>
      </div>
    </div>
  );
}