import { safeImageSource } from "./documentMedia";
import { markdownToNoteHtml, noteHtmlToMarkdown } from "./noteFormat";
import { insertNoteFigure, removeEmptyImageSections } from "./noteFigures";

export async function readLinkSource(url, signal) {
  const response = await fetch("/api/source", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }), signal });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.text?.trim()) throw new Error(data.error || "This page could not be read. Paste its text instead.");
  return data;
}
export function sourceTables(tables = []) {
  return tables.map(rows => {
    const clean = row => row.map(cell => String(cell).replace(/\|/g, "\\|").replace(/[\r\n]/g, " "));
    const header = clean(rows[0]);
    return [header, header.map(() => "---"), ...rows.slice(1).map(clean)].map(row => `| ${row.join(" | ")} |`).join("\n");
  }).join("\n\n");
}
export function notesWithSourceImages(markdown, source) {
  const images = (source?.images || []).slice(0, 6), used = new Set();
  const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(String(markdown));
  const supplied = new Set(images.filter(image => safeImageSource(image.src)).map(image => image.src));
  host.querySelectorAll("img").forEach(image => {
    if (!supplied.has(image.getAttribute("src"))) image.remove();
    else used.add(images.findIndex(item => item.src === image.getAttribute("src")));
  });
  const slots = [], walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT), nodes = [];
  while (walker.nextNode()) if (!walker.currentNode.parentElement.closest("pre,code,[data-classio-equation]")) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const pattern = /\[\[(SOURCE_IMAGE|IMAGE|PDF):\s*([^\]]+)\]\]/gi, fragment = document.createDocumentFragment(); let match, start = 0, found = false;
    while ((match = pattern.exec(node.textContent))) {
      found = true; fragment.append(node.textContent.slice(start, match.index));
      if (match[1].toUpperCase() === "SOURCE_IMAGE") { const anchor = document.createElement("span"); fragment.append(anchor); slots.push({ index: Number(match[2]) - 1, anchor }); }
      start = pattern.lastIndex;
    }
    if (found) { fragment.append(node.textContent.slice(start)); node.replaceWith(fragment); }
  }
  if (!slots.length) images.slice(0, 2).forEach((_image, index) => slots.push({ index }));
  for (const { index, anchor } of slots) {
    const image = images[index];
    if (image && !used.has(index)) {
      const figure = { ...image, source: image.source || source.url };
      if (insertNoteFigure(host, figure, image.title, anchor) || insertNoteFigure(host, figure, image.title + " " + (source.title || ""), anchor)) used.add(index);
    }
    anchor?.remove();
  }
  removeEmptyImageSections(host);
  return noteHtmlToMarkdown(host.innerHTML);
}
