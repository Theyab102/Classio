import DOMPurify from "dompurify";
import { marked } from "marked";
import TurndownService from "turndown";
import { gfm } from "turndown-plugin-gfm";
import { safeImageSource } from "./documentMedia";

const toMarkdown = new TurndownService({ headingStyle: "atx", bulletListMarker: "-", codeBlockStyle: "fenced" });
toMarkdown.use(gfm);
toMarkdown.addRule("embeddedStudyImages", {
  filter: node => node.nodeName === "IMG" && /^data:/.test(node.getAttribute("src") || ""),
  replacement: (_content, node) => `\n\n[Image: ${node.getAttribute("alt") || "Embedded figure"}]\n\n`,
});

export function sanitizeNoteHtml(html) {
  const fragment = DOMPurify.sanitize(html, {
    RETURN_DOM_FRAGMENT: true, USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "form", "video", "audio", "input", "button", "iframe", "source"],
    FORBID_ATTR: ["srcset", "id", "class"],
    ALLOW_DATA_ATTR: false,
  });
  for (const node of fragment.querySelectorAll("[style]")) {
    const alignment = node.style.textAlign;
    const family = node.style.fontFamily.replace(/["']/g,'').trim();
    const size = node.style.fontSize;
    node.removeAttribute("style");
    if (["left", "center", "right", "justify"].includes(alignment)) node.style.textAlign = alignment;
    if(['DM Sans Variable','Fraunces Variable','system-ui','monospace'].includes(family))node.style.fontFamily=family;
    if(/^\d+(?:\.\d+)?px$/.test(size)&&parseFloat(size)>=10&&parseFloat(size)<=48)node.style.fontSize=size;
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
    return safeImageSource(source) ? `![${(caption || "Study image").replace(/[\[\]]/g, "")}](<${source.replace(/[<>]/g, "")}>)` : (caption || "");
  }).replace(/^\[IMG:(https:\/\/[^\]]+)\]\s*$/gim, "![]($1)");
  return sanitizeNoteHtml(marked.parse(normalized, { gfm: true }));
}

export function noteHtmlToMarkdown(html) {
  return toMarkdown.turndown(html);
}
