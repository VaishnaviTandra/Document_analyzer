import { useState } from "react";
import { uploadPDF } from "../api";

function UploadPanel({ setIsReady }) {
  const [file, setFile] = useState(null);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      await uploadPDF(formData);
      alert("Upload successful ✅");

      setIsReady(true); // 🔥 enable chat
    } catch (err) {
      console.error(err);
      alert("Upload failed ❌");
    }
  };

  return (
    <div className="upload-panel">
      <input
        type="file"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handleUpload}>Upload PDF</button>
    </div>
  );
}

export default UploadPanel;