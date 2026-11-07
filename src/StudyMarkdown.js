import { useMemo } from "react";
import { markdownToNoteHtml } from "./noteFormat";
import { typesetMath } from "./mathMarkdown";

export default function StudyMarkdown({ text = "", className = "", allowImages = false }) {
  const html = useMemo(() => {
    const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(String(text));
    if (!allowImages) host.querySelectorAll("img").forEach(image => image.replaceWith(image.alt || ""));
    typesetMath(host);
    return host.innerHTML;
  }, [text, allowImages]);
  return <div className={`study-markdown ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
