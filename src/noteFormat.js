import DOMPurify from "dompurify";
import { marked } from "marked";
import TurndownService from "turndown";
import { gfm } from "turndown-plugin-gfm";
import { safeImageSource } from "./documentMedia";

const toMarkdown = new TurndownService({ headingStyle: "atx", bulletListMarker: "-", codeBlockStyle: "fenced" });
toMarkdown.use(gfm);
toMarkdown.addRule("studyEquations", {
  filter: node => node.nodeName === "SPAN" && node.hasAttribute("data-classio-equation"),
  replacement: (_content, node) => `$${node.getAttribute("data-latex") || ""}$`,
});
toMarkdown.addRule("embeddedStudyImages", {
  filter: node => node.nodeName === "IMG" && /^data:/.test(node.getAttribute("src") || ""),
  replacement: (_content, node) => `\n\n[Image: ${node.getAttribute("alt") || "Embedded figure"}]\n\n`,
});

export function sanitizeNoteHtml(html) {
  const fragment = DOMPurify.sanitize(html, {
    RETURN_DOM_FRAGMENT: true, USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "form", "video", "audio", "input", "button", "iframe", "source"],
    FORBID_ATTR: ["srcset", "id", "class"],
    ALLOW_DATA_ATTR: false, ADD_ATTR: ["data-classio-equation", "data-latex"],
  });
  for (const node of fragment.querySelectorAll("[style]")) {
    const alignment = node.style.textAlign;
    const font = node.style.fontFamily.replace(/['"]/g, "");
    const size = node.style.fontSize;
    node.removeAttribute("style");
    if (["left", "center", "right", "justify"].includes(alignment)) node.style.textAlign = alignment;
    if (["DM Sans Variable", "Fraunces Variable", "ui-monospace"].includes(font)) node.style.fontFamily = font;
    if ([12, 14, 16, 18, 20, 24, 28, 32].some(value => size === `${value}px`)) node.style.fontSize = size;
  }
  for (const node of fragment.querySelectorAll("[data-classio-equation],[data-latex]")) {
    if (node.tagName !== "SPAN" || !node.hasAttribute("data-classio-equation")) { node.removeAttribute("data-classio-equation"); node.removeAttribute("data-latex"); }
    else { const latex = (node.getAttribute("data-latex") || "").slice(0, 600); node.setAttribute("data-latex", latex); node.textContent = `\\(${latex}\\)`; }
  }
  for (const image of fragment.querySelectorAll("img")) {
    if (!safeImageSource(image.getAttribute("src"))) image.replaceWith(image.getAttribute("alt") || "");
    else image.setAttribute("referrerpolicy", "no-referrer");
  }
  const container = document.createElement("div");
  container.append(fragment);
  return container.innerHTML;
}

export function markdownToNoteHtml(markdown = "") {
  const normalized = markdown.replace(/(?:^|\n)TABLE_START\s*\n([\s\S]*?)\nTABLE_END(?=\n|$)/gi, (block, body) => {
    const lines = body.trim().split("\n");
    const headers = lines[0]?.match(/^headers:\s*(.+)/i)?.[1].split("|").map(value => value.trim());
    if (!headers?.length) return block;
    const rows = lines.slice(1).map(line => line.replace(/^row:\s*/i, "").split("|").map(value => value.trim()));
    return "\n\n" + [headers, headers.map(() => "---"), ...rows.map(row => headers.map((_, index) => row[index] || ""))]
      .map(row => `| ${row.join(" | ")} |`).join("\n") + "\n\n";
  }).replace(/^\[WebImage:\s*(\S+?)(?:\s*\|\s*([^\]]*))?\]\s*$/gim, (_match, source, caption) => {
    return safeImageSource(source) ? `![${(caption || "Study image").replaceAll("[", "").replaceAll("]", "")}](<${source.replace(/[<>]/g, "")}>)` : (caption || "");
  }).replace(/^\[IMG:(https:\/\/[^\]]+)\]\s*$/gim, "![]($1)");
  const host = document.createElement("div"); host.innerHTML = sanitizeNoteHtml(marked.parse(normalized, { gfm: true }));
  const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT), nodes = [];
  while (walker.nextNode()) if (!walker.currentNode.parentElement.closest("pre,code,[data-classio-equation]")) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const pattern = /\$\$([\s\S]+?)\$\$|\$([^$\n]{1,600})\$/g, text = node.textContent;
    let match, start = 0, found = false; const fragment = document.createDocumentFragment();
    while ((match = pattern.exec(text))) { found = true; fragment.append(text.slice(start, match.index)); const span = document.createElement("span"); const latex = (match[1] || match[2]).slice(0, 600); span.setAttribute("data-classio-equation", "true"); span.setAttribute("data-latex", latex); span.textContent = `\\(${latex}\\)`; fragment.append(span); start = pattern.lastIndex; }
    if (found) { fragment.append(text.slice(start)); node.replaceWith(fragment); }
  }
  return host.innerHTML;
}

export function noteHtmlToMarkdown(html) {
  return toMarkdown.turndown(html);
}
