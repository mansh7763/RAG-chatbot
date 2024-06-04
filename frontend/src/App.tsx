import React, { useState } from "react";
import UploadPDF from "../src/components/uploadPdf";
import Chatbot from "../src/components/chatBot";
const App = () => {
  const [pdfId, setPdfId] = useState(null);

  return (
    <div>
      <h1>PDF Chatbot</h1>

      <UploadPDF setPdfId={setPdfId} />
      {pdfId && <Chatbot pdfId={pdfId} />}
    </div>
  );
};

export default App;
