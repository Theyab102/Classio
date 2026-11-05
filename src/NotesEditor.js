import { useEffect, useRef, useState } from "react";
import { EditorContent, useEditor, useEditorState } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Image from "@tiptap/extension-image";
import { TableKit } from "@tiptap/extension-table";
import TextAlign from "@tiptap/extension-text-align";
import { getPdfContext, getPdfFigures, isPdfFile, likelyStudyImage, prepareNoteImage, searchStudyImages } from "./documentMedia";
import { markdownToNoteHtml, noteHtmlToMarkdown, sanitizeNoteHtml } from "./noteFormat";
import { downloadStudyFile, sendStudyRequest } from "./studyTools";
import "./documentStudio.css";

const editorExtensions = [
  StarterKit.configure({ link: { openOnClick: false, HTMLAttributes: { rel: "noopener noreferrer" } } }),
  Image.configure({ allowBase64: true, resize: { enabled: true, minWidth: 60, minHeight: 45, alwaysPreserveAspectRatio: true }, HTMLAttributes: { referrerpolicy: "no-referrer" } }),
  TableKit.configure({ table: { resizable: true, renderWrapper: true } }),
  TextAlign.configure({ types: ["heading", "paragraph"] }),
];

function insertFigure(target, image) {
  const figure = document.createElement("img");
  figure.src = image.src;
  figure.alt = image.title;
  target.before(figure);
  const caption = document.createElement("p");
  if (image.source) {
    const link = document.createElement("a");
    link.href = image.source;
    link.textContent = image.credit;
    caption.append(link);
  } else caption.textContent = image.title;
  target.before(caption);
  target.remove();
}

async function enrichDraft(markdown, pdfFile, useWeb, signal, originalHtml) {
  const root = document.createElement("div");
  root.innerHTML = markdownToNoteHtml(markdown);
  root.querySelectorAll("img").forEach(image => image.replaceWith(image.alt || ""));
  const seen = new Set();
  let figures = 0;
  for (const paragraph of [...root.querySelectorAll("p")]) {
    const match = paragraph.textContent.trim().match(/^\[\[(PDF|IMAGE):\s*([^\]]+)\]\]$/i);
    if (!match) continue;
    const [, kind, value] = match;
    const key = value.trim().toLowerCase();
    if (signal.aborted || seen.has(key) || figures >= 3) { paragraph.remove(); continue; }
    seen.add(key);
    try {
      if (kind.toUpperCase() === "PDF" && pdfFile) {
        const images = await getPdfFigures(pdfFile, Number(value));
        if (!signal.aborted && images[0]) { insertFigure(paragraph, images[0]); figures++; continue; }
      }
      if (kind.toUpperCase() === "IMAGE" && useWeb) {
        const results = await searchStudyImages(value, signal);
        const match = likelyStudyImage(value, results);
        if (match && !signal.aborted && !seen.has(match.src)) {
          seen.add(match.src);
          insertFigure(paragraph, match);
          figures++;
          continue;
        }
        paragraph.textContent = `Suggested illustration: ${value}. Review matches with the image search tool.`;
        continue;
      }
    } catch {
      paragraph.textContent = kind.toUpperCase() === "PDF" ? `Suggested source: PDF page ${value}.` : `Suggested illustration: ${value}.`;
      continue;
    }
    paragraph.remove();
  }
  if (originalHtml) {
    const original = document.createElement("div");
    original.innerHTML = originalHtml;
    const sources = new Set([...root.querySelectorAll("img")].map(image => image.getAttribute("src")));
    const originals = [...original.querySelectorAll("img")].filter(image => !sources.has(image.getAttribute("src")));
    if (originals.length) {
      const heading = document.createElement("h2");
      heading.textContent = "Original figures";
      root.append(heading, ...originals);
    }
  }
  return sanitizeNoteHtml(root.innerHTML);
}

export default function NotesEditor({ file, fileObj, getFileText, onUpdate, readSavedNotes, assistant }) {
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [topic, setTopic] = useState("");
  const [style, setStyle] = useState("Detailed");
  const [language, setLanguage] = useState("Same as source");
  const [webImages, setWebImages] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [imageTools, setImageTools] = useState(false);
  const [imageQuery, setImageQuery] = useState("");
  const [pdfPage, setPdfPage] = useState(1);
  const [searching, setSearching] = useState(false);
  const [images, setImages] = useState([]);
  const [saved, setSaved] = useState(file.namedNotes || []);
  const [versionName, setVersionName] = useState("");
  const [preview, setPreview] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const upload = useRef(null);
  const aiRequest = useRef(null);
  const mediaRequest = useRef(null);
  const [initial] = useState(() => file.notesHtml ? sanitizeNoteHtml(file.notesHtml) : markdownToNoteHtml(file.notes));
  const pdfFile = isPdfFile(fileObj) ? fileObj : null;

  const editor = useEditor({
    extensions: editorExtensions,
    content: initial,
    immediatelyRender: false,
    editorProps: {
      attributes: { class: "classio-note-content", role: "textbox", "aria-label": "Study notes", "aria-multiline": "true", spellcheck: "true", dir: "auto" },
      transformPastedHTML: sanitizeNoteHtml,
      handlePaste(_view, event) {
        const image = [...(event.clipboardData?.files || [])].find(item => item.type.startsWith("image/"));
        if (!image) return false;
        uploadImage(image);
        return true;
      },
    },
    onUpdate({ editor: current }) {
      const notesHtml = current.getHTML();
      onUpdate({ ...file, notesHtml, notes: noteHtmlToMarkdown(notesHtml) });
    },
  });
  const tools = useEditorState({
    editor,
    selector: ({ editor: current }) => ({
      table: current?.isActive("table"),
      image: current?.isActive("image"),
      bold: current?.isActive("bold"),
      italic: current?.isActive("italic"),
      underline: current?.isActive("underline"),
      strike: current?.isActive("strike"),
      bullet: current?.isActive("bulletList"),
      ordered: current?.isActive("orderedList"),
      callout: current?.isActive("blockquote"),
      heading: [1, 2, 3].find(level => current?.isActive("heading", { level })) || 0,
      outline: current?.getJSON().content?.filter(node => node.type === "heading").map(node => node.content?.map(part => part.text).join("") || "") || [],
    }),
  });
  useEffect(() => () => {
    aiRequest.current?.abort();
    mediaRequest.current?.abort();
  }, []);

  useEffect(() => {
    if (file.namedNotes || !readSavedNotes) return;
    let active = true;
    readSavedNotes().then(versions => { if (active && versions?.length) setSaved(current => current.length ? current : versions); }).catch(() => {});
    return () => { active = false; };
  }, [file.id, file.namedNotes, readSavedNotes]);

  async function uploadImage(image) {
    try {
      const src = await prepareNoteImage(image);
      editor?.chain().focus().setImage({ src, alt: image.name }).run();
    } catch (failure) { setError(failure.message); }
  }

  function addFoundImage(image) {
    const imageHtml = document.createElement("div");
    const marker = document.createElement("p");
    imageHtml.append(marker);
    insertFigure(marker, image);
    editor.chain().focus().insertContent(sanitizeNoteHtml(imageHtml.innerHTML)).run();
    setStatus(`Added ${image.title}.`);
  }

  async function findImages(source) {
    mediaRequest.current?.abort();
    const controller = new AbortController();
    mediaRequest.current = controller;
    setSearching(true);
    setError("");
    setImages([]);
    try {
      const matches = source === "pdf"
        ? await getPdfFigures(pdfFile, Number(pdfPage))
        : await searchStudyImages(imageQuery, controller.signal);
      if (!controller.signal.aborted) {
        setImages(matches);
        setStatus(matches.length ? "Review the results and insert the right image." : "No licensed image results. Try a more specific search.");
      }
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure.message);
    } finally {
      if (!controller.signal.aborted) setSearching(false);
    }
  }

  async function generate(action) {
    if (!editor || aiRequest.current) return;
    const controller = new AbortController();
    aiRequest.current = controller;
    setError("");
    setDraft("");
    setBusy(true);
    setStatus(action === "organize" ? "Organizing your writing…" : "Preparing study notes…");
    try {
      const originalHtml = editor.getHTML();
      const existing = noteHtmlToMarkdown(originalHtml);
      const source = action === "organize" || action === "table" ? existing
        : pdfFile ? await getPdfContext(pdfFile) : await getFileText();
      if (!source?.trim() && (action !== "generate" || !topic.trim())) {
        throw new Error(action === "generate" ? "Enter a topic or choose a file with readable text." : "Write a few notes before using this tool.");
      }
      const instruction = action === "table"
        ? "Return one compact GFM Markdown table based on the notes. Include clear column names and only supported information."
        : `Create ${style.toLowerCase()} study notes. Use a clear H1 title, H2 sections, concise paragraphs, lists, bold key terms and callout blockquotes. Use GFM Markdown tables for comparisons or structured facts. ${action === "organize" ? "Preserve the author's facts and order where useful. Fix wording and organization without inventing facts." : "Stay grounded in the source when provided."}`;
      const raw = await sendStudyRequest({ messages: [
        { role: "system", content: `${instruction} Language: ${language}. Return Markdown only. Do not write raw HTML or external URLs. ${pdfFile ? "For up to three relevant source figures, put [[PDF:3]] on a separate line with an actual page number from the source markers." : ""} ${webImages && action !== "table" ? "Suggest up to three precise visual searches using [[IMAGE:specific subject]] on their own lines. Skip decorative or uncertain images." : ""}` },
        { role: "user", content: `Topic: ${topic.trim() || file.name}\n\n${(source || "").slice(0, 16000)}` },
      ], maxTokens: action === "table" ? 1500 : 5500 }, controller.signal);
      const html = await enrichDraft(raw.replace(/^\s*```(?:markdown|md)?\s*\n|\n```\s*$/g, ""), pdfFile, webImages, controller.signal, action === "organize" ? originalHtml : "");
      if (!controller.signal.aborted) {
        if (action === "table") {
          const document = new DOMParser().parseFromString(html, "text/html");
          const table = document.querySelector("table");
          if (!table) throw new Error("The AI did not return a table. Try a clearer set of notes.");
          editor.chain().focus().insertContent(table.outerHTML).run();
          setStatus("Table inserted. You can edit every cell.");
        } else {
          setDraft(html);
          setStatus("Review the draft before applying it.");
        }
      }
    } catch (failure) {
      if (!controller.signal.aborted) { setError(failure.message); setStatus(""); }
    } finally {
      if (aiRequest.current === controller) {
        aiRequest.current = null;
        if (!controller.signal.aborted) setBusy(false);
      }
    }
  }

  function saveVersion(event) {
    event.preventDefault();
    const name = versionName.trim();
    if (!name || editor.isEmpty) return;
    const html = editor.getHTML();
    const notes = noteHtmlToMarkdown(html);
    const next = [{ name, html, text: notes, date: new Date().toLocaleDateString() }, ...saved.filter(item => item.name !== name)];
    setSaved(next);
    setVersionName("");
    onUpdate({ ...file, notes, notesHtml: html, namedNotes: next });
    setStatus(`Saved version: ${name}.`);
  }

  if (!editor) return <p role="status">Opening notes editor…</p>;
  const run = operation => operation(editor.chain().focus()).run();
  const action = (label, operation, pressed) => <button type="button" disabled={preview} aria-pressed={pressed} onMouseDown={event => event.preventDefault()} onClick={() => run(operation)}>{label}</button>;

  return (
    <section className="notes-studio" aria-label="Notes editor">
      <header className="studio-header">
        <div><p className="studio-eyebrow">Notebook</p><h2>Study notes</h2></div>
        <div className="studio-controls">
          {assistant && <button type="button" aria-expanded={assistantOpen} onClick={() => setAssistantOpen(value => !value)}>AI assistant</button>}
          <button type="button" onClick={() => downloadStudyFile(new Blob([`<!doctype html><meta charset="utf-8"><style>body{max-width:760px;margin:40px auto;font:16px/1.65 system-ui}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.45rem}img{max-width:100%}</style>${sanitizeNoteHtml(editor.getHTML())}`], { type: "text/html" }), "classio-notes.html")}>Download notes</button>
          {file.transcript && <button type="button" onClick={() => downloadStudyFile(new Blob([file.transcript], { type: "text/plain" }), "classio-transcript.txt")}>Download transcript</button>}
        </div>
      </header>
      <div className="notes-studio-grid">
        <article className="note-workspace">
          <div className="studio-controls note-toolbar" role="toolbar" aria-label="Note formatting">
            <select aria-label="Paragraph style" disabled={preview} value={tools?.heading || 0} onChange={event => {
              const level = Number(event.target.value);
              run(chain => level ? chain.setHeading({ level }) : chain.setParagraph());
            }}>
              <option value="0">Paragraph</option><option value="1">Title</option><option value="2">Heading</option><option value="3">Subheading</option>
            </select>
            {action("Bold", chain => chain.toggleBold(), tools?.bold)}
            {action("Italic", chain => chain.toggleItalic(), tools?.italic)}
            {action("Underline", chain => chain.toggleUnderline(), tools?.underline)}
            {action("Strike", chain => chain.toggleStrike(), tools?.strike)}
            {action("Bullets", chain => chain.toggleBulletList(), tools?.bullet)}
            {action("Numbered list", chain => chain.toggleOrderedList(), tools?.ordered)}
            {action("Callout", chain => chain.toggleBlockquote(), tools?.callout)}
            <select aria-label="Align text" disabled={preview} value="" onChange={event => run(chain => chain.setTextAlign(event.target.value))}>
              <option value="" disabled>Align</option><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option><option value="justify">Justify</option>
            </select>
            {action("Insert table", chain => chain.insertTable({ rows: 3, cols: 3, withHeaderRow: true }))}
            <button type="button" disabled={preview} onMouseDown={event => event.preventDefault()} onClick={() => upload.current?.click()}>Upload image</button>
            <input ref={upload} type="file" accept="image/png,image/jpeg,image/webp,image/gif" hidden onChange={event => {
              const file = event.target.files[0];
              event.target.value = "";
              if (file) uploadImage(file);
            }} />
            <button type="button" disabled={preview} aria-expanded={imageTools} onClick={() => setImageTools(value => !value)}>Find images</button>
            {action("Undo", chain => chain.undo())}
            {action("Redo", chain => chain.redo())}
            <button type="button" aria-pressed={preview} onClick={() => { editor.setEditable(preview); setPreview(value => !value); if (!preview) setImageTools(false); }}>{preview ? "Edit notes" : "Reading view"}</button>
          </div>
          {tools?.table && <div className="studio-controls note-table-tools" aria-label="Table actions">
            {action("Add row", chain => chain.addRowAfter())}{action("Add column", chain => chain.addColumnAfter())}
            {action("Delete row", chain => chain.deleteRow())}{action("Delete column", chain => chain.deleteColumn())}
            {action("Merge cells", chain => chain.mergeCells())}{action("Split cell", chain => chain.splitCell())}
            {action("Delete table", chain => chain.deleteTable())}
          </div>}
          {tools?.image && !preview && <div className="studio-controls note-table-tools">
            <button type="button" onClick={() => run(chain => chain.deleteSelection())}>Remove selected image</button>
          </div>}
          {imageTools && (
            <div className="note-image-tools">
              <form className="studio-controls" onSubmit={event => { event.preventDefault(); findImages("web"); }}>
                <input aria-label="Image search terms" placeholder="Example: plant cell diagram" value={imageQuery} onChange={event => setImageQuery(event.target.value)} />
                <button type="submit" disabled={searching || !imageQuery.trim()}>Search Wikimedia</button>
                {imageQuery.trim() && <a href={`https://www.google.com/search?tbm=isch&q=${encodeURIComponent(imageQuery.trim())}`} target="_blank" rel="noreferrer">Search Google Images</a>}
              </form>
              {pdfFile && <div className="studio-controls">
                <label>PDF page <input type="number" min="1" value={pdfPage} onChange={event => setPdfPage(event.target.value)} /></label>
                <button type="button" disabled={searching} onClick={() => findImages("pdf")}>Find figures on page</button>
              </div>}
              {searching && <p role="status">Finding images…</p>}
              <div className="note-image-results">
                {images.map(image => <figure key={image.src}>
                  <img src={image.src} alt={image.title} referrerPolicy="no-referrer" loading="lazy" />
                  <figcaption>{image.source ? <a href={image.source} target="_blank" rel="noreferrer">{image.credit}</a> : image.title}</figcaption>
                  <button type="button" onClick={() => addFoundImage(image)}>Insert image</button>
                </figure>)}
              </div>
            </div>
          )}
          <div className="note-paper"><EditorContent editor={editor} /></div>
          <form className="studio-controls note-versions" onSubmit={saveVersion}>
            <input aria-label="Version name" placeholder="Version name" value={versionName} onChange={event => setVersionName(event.target.value)} />
            <button type="submit" disabled={!versionName.trim()}>Save named version</button>
            {saved.length > 0 && <select aria-label="Load a saved version" value="" onChange={event => {
              const version = saved[Number(event.target.value)];
              if (version) editor.commands.setContent(version.html ? sanitizeNoteHtml(version.html) : markdownToNoteHtml(version.text));
            }}><option value="" disabled>Load version</option>{saved.map((version, index) => <option key={version.name} value={index}>{version.name}</option>)}</select>}
          </form>
        </article>
        <aside className="notes-studio-side">
          <section>
            <h3>Improve with AI</h3>
            <p>Write freely, then let AI organize your ideas or build a comparison table.</p>
            <div className="studio-controls">
              <button type="button" disabled={busy} onClick={() => generate("organize")}>Organize my notes</button>
              <button type="button" disabled={busy} onClick={() => generate("table")}>Make a table</button>
            </div>
            <label>Topic <input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with a study file" /></label>
            <label>Note style <select value={style} onChange={event => setStyle(event.target.value)}><option>Detailed</option><option>Exam review</option><option>Concise summary</option><option>Simple explanations</option></select></label>
            <label>Language <select value={language} onChange={event => setLanguage(event.target.value)}><option>Same as source</option><option>English</option><option>Arabic</option><option>Chinese</option><option>French</option><option>Spanish</option></select></label>
            <label className="studio-check"><input type="checkbox" checked={webImages} onChange={event => setWebImages(event.target.checked)} /> Look for relevant web images</label>
            <div className="studio-controls"><button type="button" disabled={busy} onClick={() => generate("generate")}>{busy ? "Working…" : "Generate notes"}</button></div>
          </section>
          {!!tools?.outline?.length && <section><h3>On this page</h3><ol>{tools.outline.map((heading, index) => <li key={index}><button type="button" onClick={() => editor.view.dom.querySelectorAll("h1,h2,h3")[index]?.scrollIntoView({ block: "nearest" })}>{heading || "Untitled heading"}</button></li>)}</ol></section>}
        </aside>
      </div>
      {assistantOpen && <div className="note-assistant-slot">{assistant(noteHtmlToMarkdown(editor.getHTML()))}</div>}
      {status && <p className="studio-status" role="status">{status}</p>}
      {error && <p className="studio-error" role="alert">{error}</p>}
      {draft && <section className="note-draft" aria-label="AI draft">
        <h3>AI draft</h3>
        <div className="classio-note-content" dangerouslySetInnerHTML={{ __html: draft }} />
        <div className="studio-controls">
          <button type="button" onClick={() => {
            editor.commands.setContent(draft);
            setDraft("");
            setStatus("Draft applied. Use Undo to return to your original notes.");
          }}>Apply to notes</button>
          <button type="button" onClick={() => { editor.chain().focus("end").insertContent(draft).run(); setDraft(""); }}>Append to notes</button>
          <button type="button" onClick={() => { setDraft(""); setStatus("Draft discarded."); }}>Discard draft</button>
        </div>
      </section>}
    </section>
  );
}
