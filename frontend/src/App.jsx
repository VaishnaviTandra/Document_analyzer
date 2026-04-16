import { useState } from "react";
import Navbar from "./components/Navbar";
// import UploadPanel from "./components/UploadPanel";
import ChatBox from "./components/ChatBox";
import "./styles.css";

function App() {
  const [isReady, setIsReady] = useState(false);

  return (
    <div>
      <Navbar />
      {/* <UploadPanel setIsReady={setIsReady} /> */}
      <ChatBox isReady={isReady} />
    </div>
  );
}

export default App;