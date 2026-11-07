import { useState } from "react";
import UiIcon from "./UiIcon";

export default function InlineAssistant({ onAsk, contextLabel = "your study material" }) {
  const [text, setText] = useState("");
  return <form className="inline-assistant" aria-label="Ask Classio" onSubmit={event => { event.preventDefault(); if (text.trim()) { onAsk(text.trim()); setText(""); } }}>
    <span className="assistant-mark"><UiIcon name="generate" size={22} /></span>
    <label><span>Study with Classio</span><input aria-label="Ask Classio anything" placeholder="Ask Classio anything..." value={text} onChange={event => setText(event.target.value)} maxLength={8000} /></label>
    <button type="submit" className="quiet-button is-primary" disabled={!text.trim()} aria-label={`Ask about ${contextLabel}`}><UiIcon name="send" /><span>Ask</span></button>
  </form>;
}
