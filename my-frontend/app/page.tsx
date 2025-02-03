"use client";  // enable client-side features like hooks or local component state

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";

// This is a simple Next.js 13 App Router page that mirrors your previous index.js page functionality.
export default function Home() {
  // Hard-coded sessionId for simplicity
  const [sessionId, setSessionId] = useState("my-session-id");
  
  // Entire chat history from the backend
  const [chatHistory, setChatHistory] = useState<string[]>([]);

  // Track user input
  const [userInput, setUserInput] = useState("");

  // Sidebar width dedicated to session ID
  const sidebarWidth = "200px";

  // Ref for the scrollable container in the right pane
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  // Ref for the dummy div at the bottom of the chat for scrolling
  const bottomRef = useRef<HTMLDivElement>(null);

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

  // Helper to call your /history endpoint
  async function getHistory() {
    try {
      const response = await fetch("http://localhost:8000/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      // data.response should be an array of chat messages
      setChatHistory(data.response);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  }

  // Helper to call your /chat endpoint without streaming, ensuring proper order.
  async function sendMessage() {
    if (!userInput.trim()) return;

    // Save the current user message
    const message = userInput;

    // Append the user message immediately.
    setChatHistory((prev) => [...prev, `User: ${message}`]);
    
    // Clear the input field immediately.
    setUserInput("");

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_input: message,
          session_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();

      // Append the AI response after the user message.
      setChatHistory((prev) => [...prev, `AI: ${data.response}`]);
    } catch (err) {
      console.error("Failed to send message:", err);
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
    if (text.startsWith("User:")) {
      return text.replace(/^User:\s*/, "");
    }
    if (text.startsWith("AI:")) {
      return text.replace(/^AI:\s*/, "");
    }
    return text;
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
          overflow: "hidden",
        }}
      >
        <h3>Session</h3>
        <div style={{ marginBottom: "1rem" }}>
          <label htmlFor="sessionId" style={{ display: "block" }}>
            Session ID:
          </label>
          <input
            id="sessionId"
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            style={{
              width: "100%",
              padding: "0.5rem",
              boxSizing: "border-box",
            }}
          />
        </div>
        <button onClick={getHistory} style={{ width: "100%" }}>
          Refresh History
        </button>
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
          <h1>My AI Chat</h1>
          <div style={{ marginBottom: "1rem" }}>
            <h2>Chat History</h2>
            <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {chatHistory.map((text, index) => (
                <li key={index} style={{ marginBottom: "0.5rem" }}>
                  <div
                    style={{
                      display: "inline-block",
                      background: text.startsWith("User:")
                        ? "#f0f0f0"
                        : "#ffffff",
                      padding: "0.5rem",
                      borderRadius: "4px",
                      wordBreak: "break-word",
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
              width: "33.33vw", // Approximately 1/3 of the full screen width
              padding: "0.5rem",
              boxSizing: "border-box",
            }}
          />
        </div>
      </div>
    </div>
  );
}