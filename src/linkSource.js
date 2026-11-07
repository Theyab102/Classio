import { safeImageSource } from "./documentMedia";
import { markdownToNoteHtml, noteHtmlToMarkdown } from "./noteFormat";

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
  const figure = index => {
    const image = images[index]; if (!image || !safeImageSource(image.src) || used.has(index)) return "";
    used.add(index);
    const alt = (image.title || "Source figure").replaceAll("[", " ").replaceAll("]", " ").replace(/[<>\r\n]/g, " ").slice(0, 250);
    return `\n\n![${alt}](<${image.src.replace(/[<>]/g, "")}>)\n\n[Figure from the source page](<${(image.source || source.url).replace(/[<>]/g, "")}>)\n\n`;
  };
  let text = String(markdown).replace(/^\s*\[\[SOURCE_IMAGE:\s*(\d+)\]\]\s*$/gim, (_match, n) => figure(Number(n) - 1));
  // A provider can omit figure markers; the real source images remain available
  // as a clearly labelled appendix instead of being replaced with invented art.
  if (images.some(image => safeImageSource(image.src)) && !used.size) text += "\n\n## Source figures\n" + images.map((image, i) => ({ image, i })).filter(({ image }) => safeImageSource(image.src)).slice(0, 2).map(({ i }) => figure(i)).join("\n");
  const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(text);
  const supplied = new Set(images.filter(image => safeImageSource(image.src)).map(image => image.src));
  host.querySelectorAll("img").forEach(image => {
    if (!supplied.has(image.getAttribute("src"))) image.remove();
  });
  return noteHtmlToMarkdown(host.innerHTML);
}
