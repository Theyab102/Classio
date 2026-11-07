import { acknowledgeStudyImage } from "./documentMedia";
import { markdownToNoteHtml, sanitizeNoteHtml } from "./noteFormat";
import { getPdfFigures, getPresentationFigures, likelyStudyImage, prepareSourceFigure, safeImageSource, searchStudyImages } from "./documentMedia";

const generic = new Set(["the", "and", "for", "with", "from", "image", "figure", "diagram", "picture", "illustration", "search", "suggested", "visual", "source", "labeled", "labelled", "specific", "subject"]);
export const subjectWords = text => [...new Set((String(text || "").toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || []).map(word => word.replace(/s$/, "")).filter(word => !generic.has(word)))];
const imageHeading = text => /(?:suggested|visual|image|figure).*(?:search|suggestion)|source figures/i.test(text);
const topBlock = (root, node) => { let item = node?.nodeType === 1 ? node : node?.parentElement; while (item && item.parentElement !== root) item = item.parentElement; return item; };

export function imagePlacement(root, query, anchor) {
  const terms = subjectWords(query), headings = [...root.querySelectorAll("h1,h2,h3,h4")].filter(node => !imageHeading(node.textContent));
  let best, bestScore = 0;
  for (const heading of headings) {
    const header = subjectWords(heading.textContent), content = []; let next = heading.nextElementSibling;
    while (next && !/^H[1-4]$/.test(next.tagName)) { content.push(next.textContent); next = next.nextElementSibling; }
    const body = subjectWords(content.join(" ").slice(0, 1800));
    const score = terms.reduce((sum, word) => sum + (header.includes(word) ? 4 : body.includes(word) ? 1 : 0), 0);
    if (score > bestScore) { best = heading; bestScore = score; }
  }
  if (bestScore) {
    let first = best.nextElementSibling;
    while (first && !/^H[1-4]$/.test(first.tagName)) {
      if (first.tagName === "P" && first.textContent.trim().length > 10 && !first.querySelector("img")) return first;
      first = first.nextElementSibling;
    }
    return best;
  }
  const block = topBlock(root, anchor);
  if (block) {
    let previous = block.previousElementSibling;
    while (previous && !/^H[1-4]$/.test(previous.tagName)) previous = previous.previousElementSibling;
    if (!previous || !imageHeading(previous.textContent)) return block;
  }
  if (!headings.length || headings.length === 1) return [...root.children].find(node => node.tagName === "P" && node.textContent.length > 10) || headings[0];
  return null;
}

export function insertNoteFigure(root, image, query, anchor) {
  if (!safeImageSource(image?.src)) return false;
  const target = imagePlacement(root, query || image.title, anchor); if (!target) return false;
  const photo = document.createElement("img"); photo.src = image.src; photo.alt = image.title || query || "Study figure"; photo.setAttribute("referrerpolicy", "no-referrer");
  const caption = document.createElement("p");
  if (/^https:\/\//i.test(image.source || "")) { const link = document.createElement("a"); link.href = image.source; link.textContent = image.credit || "Figure from the source page"; caption.append(link); }
  else caption.textContent = image.credit || image.title || "Source figure";
  target.after(photo, caption); return true;
}

export function removeEmptyImageSections(root) {
  root.querySelectorAll("p,li").forEach(node => { if (!node.textContent.trim() && !node.querySelector("img,[data-classio-equation]")) node.remove(); });
  root.querySelectorAll("h1,h2,h3,h4").forEach(heading => {
    if (!imageHeading(heading.textContent)) return;
    let next = heading.nextElementSibling;
    if (!next || /^H[1-4]$/.test(next.tagName) || !next.textContent.trim()) heading.remove();
  });
}

export async function enrichNoteImages(markdown, { pdfFile, fileObj, useWeb, signal, originalHtml = "", existingHtml, search = searchStudyImages } = {}) {
  const root = document.createElement("div"); root.innerHTML = existingHtml === undefined ? markdownToNoteHtml(markdown) : sanitizeNoteHtml(existingHtml);
  if (existingHtml === undefined) root.querySelectorAll("img").forEach(image => image.replaceWith(image.alt || ""));
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), texts = [];
  while (walker.nextNode()) if (!walker.currentNode.parentElement.closest("pre,code,[data-classio-equation]")) texts.push(walker.currentNode);
  const markers = [];
  for (const node of texts) {
    const pattern = /\[\[(PDF|IMAGE):\s*([^\]]+)\]\]/gi, fragment = document.createDocumentFragment(); let match, start = 0, found = false;
    while ((match = pattern.exec(node.textContent))) {
      found = true; fragment.append(node.textContent.slice(start, match.index));
      const anchor = document.createElement("span"); fragment.append(anchor);
      markers.push({ kind: match[1].toUpperCase(), query: match[2].trim(), anchor }); start = pattern.lastIndex;
    }
    if (found) { fragment.append(node.textContent.slice(start)); node.replaceWith(fragment); }
  }
  const seen = new Set([...root.querySelectorAll("img")].map(image => image.getAttribute("src"))); let added = 0, missing = 0, sourceFigures;
  const requests = markers.slice(0, 3).map(async marker => {
    try {
      if (signal?.aborted) return {};
      if (marker.kind === "PDF" && pdfFile) return { marker, image: (await getPdfFigures(pdfFile, Number(marker.query)))[0] };
      if (!useWeb) return { marker };
      if (fileObj && /\.pptx$/i.test(fileObj.name || "")) {
        sourceFigures ||= getPresentationFigures(fileObj);
        const source = likelyStudyImage(marker.query, await sourceFigures); if (source) return { marker, image: await prepareSourceFigure(source) };
      }
      const controller = new AbortController(), cancel = () => controller.abort();
      signal?.addEventListener("abort", cancel, { once: true });
      const timer = setTimeout(cancel, 8000); let results;
      try { results = await search(marker.query, controller.signal); }
      finally { clearTimeout(timer); signal?.removeEventListener("abort", cancel); }
      return { marker, image: likelyStudyImage(marker.query, results) };
    } catch { return { marker }; }
  });
  for (let { marker, image } of await Promise.all(requests)) {
    if (!marker || signal?.aborted) continue;
    if (image && seen.has(image.src)) continue;
    if(image)try{await acknowledgeStudyImage(image,signal);}catch{image=null;}
    if (!image || !insertNoteFigure(root, image, marker.query, marker.anchor)) missing++;
    else { seen.add(image.src); added++; }
  }
  markers.forEach(marker => marker.anchor.remove()); removeEmptyImageSections(root);
  if (originalHtml) {
    const original = document.createElement("div"); original.innerHTML = originalHtml;
    original.querySelectorAll("img").forEach(image => {
      if (!seen.has(image.getAttribute("src"))) {
        if (!insertNoteFigure(root, { src: image.getAttribute("src"), title: image.alt }, image.alt)) root.append(image.cloneNode(true));
      }
    });
  }
  return { html: sanitizeNoteHtml(root.innerHTML), added, missing };
}
