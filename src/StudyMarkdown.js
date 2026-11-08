import { useMemo } from "react";
import { markdownToNoteHtml } from "./noteFormat";
import { typesetMath } from "./mathMarkdown";

export default function StudyMarkdown({ text = "", className = "", allowImages = false, inline = false }) {
  const html = useMemo(() => {
    const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(String(text));
    if (!allowImages) host.querySelectorAll("img").forEach(image => image.replaceWith(image.alt || ""));
    typesetMath(host);
    if(inline) host.querySelectorAll("p,h1,h2,h3,h4,h5,h6,blockquote,ul,ol,li").forEach(node => { const span=document.createElement("span"); span.append(...node.childNodes); node.replaceWith(span,document.createTextNode(" ")); });
    return host.innerHTML;
  }, [text, allowImages, inline]);
  const Tag=inline?"span":"div";
  return <Tag className={`study-markdown ${inline?"study-inline":""} ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
