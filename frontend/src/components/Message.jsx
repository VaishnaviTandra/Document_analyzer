function Message({ msg }) {
  // 🔥 Split into readable lines
  const formattedText = msg.text
    ?.split("\n")
    .filter(line => line.trim() !== "");

  return (
    <div className={`message ${msg.type}`}>

      {/* 💬 Answer */}
      <div className="message-text">
        {formattedText.map((line, i) =>
          line.startsWith("-") || line.startsWith("•") ? (
            <li key={i}>{line}</li>
          ) : (
            <p key={i}>{line}</p>
          )
        )}
      </div>

      {/* 📚 Sources */}
      {msg.sources && msg.sources.length > 0 && (
        <div className="sources">
          <strong>Sources:</strong>

          {/* 🔥 remove duplicates */}
          {[...new Map(msg.sources.map(s => [JSON.stringify(s), s])).values()]
            .map((s, i) => {

              if (s.type === "pdf") {
                return (
                  <div key={i} className="source-item">
                    📄 PDF — Page {s.page ?? "N/A"}
                  </div>
                );
              }

              if (s.type === "web") {
                return (
                  <div key={i} className="source-item">
                    🌐 Web — {s.source}
                  </div>
                );
              }

             if (s.type === "youtube") {
  const seconds = s.start || 0;
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;

  const time = `${minutes}:${secs.toString().padStart(2, "0")}`;

  const link = `https://www.youtube.com/watch?v=${s.source}&t=${seconds}s`;

  return (
    <div key={i} className="source-item">
      ▶️ YouTube —{" "}
      <a href={link} target="_blank" rel="noopener noreferrer">
        {time}
      </a>
    </div>
  );
}

              return (
                <div key={i} className="source-item">
                  Unknown Source
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}

export default Message;