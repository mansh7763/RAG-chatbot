import React, { useState } from "react";
import axios from "axios";
interface ChatbotProps {
  pdfId: string;
}
const Chatbot = ({ pdfId }: { pdfId: ChatbotProps }) => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [error, setError] = useState("");

  const handleQueryChange = (event: {
    target: { value: React.SetStateAction<string> };
  }) => {
    setQuery(event.target.value);
  };

  const handleQuery = async () => {
    try {
      const result = await axios.post("http://127.0.0.1:5000/query", {
        query: query,
        pdf_id: pdfId,
      });
      setResponse(result.data.response);
    } catch (error) {
      setError("Failed to fetch response.");
    }
  };

  return (
    <div>
      <h2>Query PDF</h2>
      <input
        type="text"
        value={query}
        onChange={handleQueryChange}
        placeholder="Enter your query"
      />
      <button onClick={handleQuery}>Query</button>
      {response && <p>Response: {response}</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default Chatbot;
