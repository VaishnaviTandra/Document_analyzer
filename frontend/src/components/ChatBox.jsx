import { useState } from "react";
import axios from "axios";
import Message from "./Message";
import Loader from "./Loader";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

function ChatBox() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState("");
  const [youtube, setYoutube] = useState("");
  const [loading, setLoading] = useState(false);

  const sendQuery = async () => {
    if (!query.trim() && !file && !url && !youtube) return;

    console.log("📄 FILE:", file); // 🔥 DEBUG

    const userMsg = {
      type: "user",
      text: query || "📎 Input provided"
    };

    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const formData = new FormData();

      // 🧠 Always send query
      formData.append("query", query || "");

      // 📄 File (ONLY if valid)
      if (file instanceof File) {
        formData.append("file", file);
      }

      // 🌐 URL
      if (url && url.trim() !== "") {
        formData.append("url", url.trim());
      }

      // ▶️ YouTube
      if (youtube && youtube.trim() !== "") {
        formData.append("youtube", youtube.trim());
      }

      const res = await API.post("/ask-all", formData, {
        headers: {
          "Content-Type": "multipart/form-data"
        }
      });

      const botMsg = {
        type: "bot",
        text: res.data.answer,
        sources: res.data.sources
      };

      setMessages((prev) => [...prev, botMsg]);

    } catch (err) {
      console.error(err);

      setMessages((prev) => [
        ...prev,
        {
          type: "bot",
          text:
            err.response?.data?.detail ||
            err.response?.data?.answer ||
            "Something went wrong ❌"
        }
      ]);
    }

    setLoading(false);
    setQuery("");

    // ❗ DO NOT clear file immediately (important)
    // setFile(null); ❌ remove this

    setUrl("");
    setYoutube("");
  };

  return (
    <div className="chat-container">

      {/* 💬 Chat */}
      <div className="chat-box">
        {messages.map((m, i) => (
          <Message key={i} msg={m} />
        ))}
        {loading && <Loader />}
      </div>

      {/* ⌨️ Input */}
      <div className="input-box">

        {/* 📄 File */}
        <input
          type="file"
          name="file"
          onChange={(e) => {
            const selectedFile = e.target.files[0];
            console.log("Selected file:", selectedFile); // 🔥 DEBUG
            setFile(selectedFile);
          }}
        />

        {/* 🌐 Web URL */}
        <input
          type="text"
          placeholder="Paste website URL..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />

        {/* ▶️ YouTube */}
        <input
          type="text"
          placeholder="Paste YouTube link..."
          value={youtube}
          onChange={(e) => setYoutube(e.target.value)}
        />

        {/* 💬 Query */}
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask your question..."
          onKeyDown={(e) => e.key === "Enter" && sendQuery()}
        />

        <button onClick={sendQuery} disabled={loading}>
          Send
        </button>
      </div>
    </div>
  );
}

export default ChatBox;