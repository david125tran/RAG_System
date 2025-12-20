// src/App.js

import { useState, useEffect } from "react";
import "./App.css";

// ------------------------------------ Model ------------------------------------
const model = {
  name: "Sandy",
  initMessage: "Hello!",
};

// ------------------------------------ Helpers ------------------------------------
const STORAGE_KEY = "chat-storage";

const buildInitialChat = () => [
  {
    role: "assistant",
    text: model.initMessage,
    timestamp: Date.now(),
  },
];

// ------------------------------------ Component ------------------------------------
export default function ChatPage() {
  const [chats, setChats] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) return JSON.parse(saved);
    } catch (err) {
      console.error("Failed to load chats", err);
    }
    return buildInitialChat();
  });

  // Persist chats
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  }, [chats]);

  return (
    <div className="chat-page">
      <h1>RAG Chat</h1>

      <ul>
        {chats.map((msg, i) => (
          <li key={i}>
            <strong>{msg.role}:</strong> {msg.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
