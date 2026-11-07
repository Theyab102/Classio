import { useMemo } from "react";
import katex from "katex";
import { markdownToNoteHtml } from "./noteFormat";

export default function StudyMarkdown({ text = "", className = "" }) {
  const html = useMemo(() => {
    const host = document.createElement("div");
    host.innerHTML = markdownToNoteHtml(String(text));
    // Provider text cannot cause an unsolicited external image request.
    host.querySelectorAll("img").forEach(image => image.replaceWith(image.alt || ""));
    host.querySelectorAll("[data-classio-equation]").forEach(node => { node.innerHTML = katex.renderToString(node.getAttribute("data-latex") || "", { throwOnError: false, trust: false, maxExpand: 500 }); });
    const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT), nodes = [];
    while (walker.nextNode()) if (!walker.currentNode.parentElement.closest("code,pre,[data-classio-equation]")) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const pattern = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\$([^$\n]{1,500})\$|\\\(([^\n]{1,500}?)\\\)/g;
      const value = node.textContent;
      let match, index = 0, found = false;
      const fragment = document.createDocumentFragment();
      while ((match = pattern.exec(value))) {
        found = true;
        fragment.append(value.slice(index, match.index));
        const span = document.createElement("span");
        span.className = "study-equation";
        span.innerHTML = katex.renderToString(match[1] || match[2] || match[3] || match[4], { throwOnError: false, trust: false, maxExpand: 500, displayMode: Boolean(match[1] || match[2]) });
        fragment.append(span);
        index = pattern.lastIndex;
      }
      if (found) { fragment.append(value.slice(index)); node.replaceWith(fragment); }
    }
    return host.innerHTML;
  }, [text]);
  return <div className={`study-markdown ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
