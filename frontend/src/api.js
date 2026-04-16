import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000"
});

// 🔥 SINGLE UNIFIED CALL (BEST PRACTICE)
export const askAll = ({ query, file, url, youtube }) => {
  const formData = new FormData();

  // 🧠 Always send query (even empty string is fine)
  formData.append("query", query || "");

  // 📄 File
  if (file) {
    formData.append("file", file);
  }

  // 🌐 Web URL
  if (url) {
    formData.append("url", url);
  }

  // ▶️ YouTube link
  if (youtube) {
    formData.append("youtube", youtube);
  }

  return API.post("/ask-all", formData, {
    headers: {
      "Content-Type": "multipart/form-data"
    }
  });
};