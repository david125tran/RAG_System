import React from "react";
import { useEffect, useMemo, useState } from "react";
import "./App.css";

// Store chat history
const STORAGE_KEY = "chat-storage";

// Point to the FastAPI that the Python backend works on
const API_URL = "http://localhost:8000/api/chat";

// Profile Picture
const avatar = "/avatar.png";
const name = "Infectious Disease Assistant";

// Render App
export default function App() {
  // ----------------- Initialize Greeting -----------------
  const initialChats = useMemo(
    () => [
      {
        role: "assistant",
        text: "Welcome!  I'm an expert in infectious diseases, feel free to ask me anything about them. 😊",
        timestamp: Date.now(),
      },
    ],
    []
  );
  

  // ----------------- Initialize Chat State (Restore from localStorage if Possible) -----------------
  const [chats, setChats] = useState(() => {
    // If there is a saved chat, return that
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      // If there is a saved chat, parse it
      if (saved) return JSON.parse(saved);
    } catch {
      // ignore
    }
    // Else, initialize a new chat
    return initialChats;
  });

  // What is currently in the input textbox
  const [input, setInput] = useState("");
  
  // True while a LLM is responding.  Used to lock the input box and show 'Thinking...'
  const [isWaiting, setIsWaiting] = useState(false);

  // Persist chat history to local storage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);


  // ----------------- Functions -----------------
  // Convert the UI chat's history into the backend-accepted format.
  // The backend supports:
  //  - React-style: [{ from: "user"|"bot", text: "..." }, ...]
  //  - Bedrock-style: [{ role: "user"|"assistant", content: "..." }, ...]
  function buildBackendHistory(allChats) {
    // (1) Exclude any "Thinking..." placeholder message we may have been added.
    const filtered = (allChats || []).filter((m) => m.role !== "thinking");

    // (2) Exclude the initial assistant greeting
    const withoutGreeting = filtered.filter(
      (m, idx) => !(idx === 0 && m.role === "assistant")
    );

    // (3) Map the roles
    return withoutGreeting.map((m) => ({
      from: m.role === "user" ? "user" : "bot",
      text: m.text,
    }));
  }

  async function sendMessage(text) {
    // (1) Don't send empty messages 
    const trimmed = (text || "").trim();
    if (!trimmed) return;

    // (2) Force user to wait for LLM response
    if (isWaiting) return;

    // (3) Append user's message immediately
    const userMsg = { role: "user", text: trimmed, timestamp: Date.now() };
    setChats((prev) => [...prev, userMsg]);
    setInput("");
    setIsWaiting(true);

    // (4) Add a temporary "thinking" bubble
    const thinkingMsg = {
      role: "thinking",
      text: "Thinking…",
      timestamp: Date.now(),
    };
    setChats((prev) => [...prev, thinkingMsg]);

    // (5) Build chat history payload
    try {
      // (5.1) Build history including the new user message, but excluding the thinking bubble
      const historyForBackend = buildBackendHistory([...chats, userMsg]);

      // (5.2) Send request as multipart/form-data
      const form = new FormData();
      form.append("message", trimmed);
      form.append("history", JSON.stringify(historyForBackend));

      // (5.3) Match the FastAPI backend signature:
      //    message: str = Form("")
      //    history: str = Form("[]")
      const res = await fetch(API_URL, {
        method: "POST",
        body: form,
      });

      // (5.4) Handle non-200 responses 
      if (!res.ok) {
        // Extract the error message 
        let detail = `Request failed (${res.status})`;
        try {
          const errJson = await res.json();
          if (errJson?.detail) detail = String(errJson.detail);
        } catch {
          // Ignore parse errors
        }
        throw new Error(detail);
      }

      // (5.5) Parse and display LLM reply
      const data = await res.json(); // { reply: "..." }
      const replyText = (data && data.reply) || "Sorry - no reply received.";

      // (5.6) Replace the thinking bubble with the assistant reply
      setChats((prev) => {
        const withoutThinking = prev.filter((m) => m.role !== "thinking");
        return [
          ...withoutThinking,
          { role: "assistant", text: replyText, timestamp: Date.now() },
        ];
      });
      // (5.7) Error handling, if the server fails.
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Unknown error occurred.";

      setChats((prev) => {
        // (5.7.1) Remove the thinking bubble
        const withoutThinking = prev.filter((m) => m.role !== "thinking");
        // (5.7.2) Explain the error
        return [
          ...withoutThinking,
          {
            role: "assistant",
            text: `⚠️ Sorry - I couldn't reach the server. ${msg}`,
            timestamp: Date.now(),
          },
        ];
      });
    // (6) Unlock the input text box
    } finally {
      setIsWaiting(false);
    }
  }

  function clearChat() {
    // Wipe persistent storage
    localStorage.removeItem(STORAGE_KEY);
    // Reset chat to initial message
    setChats(initialChats);
    // Unlock user input text box
    setIsWaiting(false);
  }

  // Render UI
  return (
    <div className="app">
      <div className="chatWidget">
        <div className="chatHeader">
          <div className="chatHeaderRow">
            <div className="avatar">
              <img src={avatar} alt={name} />
            </div>
            <div className="headerText">
              <div className="headerTitle">Need help?</div>
              <div className="headerSub">
                <span className="statusDot" /> {isWaiting ? "Thinking…" : "Online"}
              </div>
            </div>
          </div>
        </div>

        <div className="chatBody">
          <div className="messageGroup">
            {chats.map((msg, i) => (
              <div key={i} className={`msgRow ${msg.role}`}>
                {msg.role === "assistant" || msg.role === "thinking" ? (
                  <div className="msgAvatar" />
                ) : null}

                <div className={`bubble ${msg.role}`}>
                  {msg.text}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="chatFooter">
          <div className="inputRow">
            <div className="inputWrap">
              <input
                className="textInput"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={isWaiting ? "Waiting for response..." : "Type your message here..."}
                disabled={isWaiting}
                onKeyDown={(e) => {
                  if (e.key === "Enter") sendMessage(input);
                }}
              />
            </div>

            <button
              className="sendBtn"
              onClick={() => sendMessage(input)}
              aria-label="send"
              disabled={isWaiting}
              title={isWaiting ? "Waiting for response..." : "Send"}
            >
              ➤
            </button>

            <button
              className="clearChatBtn"
              onClick={() => clearChat()}
              aria-label="clear"
              disabled={isWaiting}
              title={isWaiting ? "Waiting for response..." : "Clear chat"}
            >
              🗑️
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
