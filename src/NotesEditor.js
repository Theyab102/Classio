import { useEffect, useRef, useState } from "react";
import { EditorContent, useEditor, useEditorState } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Image from "@tiptap/extension-image";
import { TableKit } from "@tiptap/extension-table";
import TextAlign from "@tiptap/extension-text-align";
import { closeHistory } from "@tiptap/pm/history";
import { TextStyleKit } from "@tiptap/extension-text-style";
import LearnMode from "./LearnMode";
import { NoteEquation } from "./NoteEquation";
import katex from "katex";
import { useStudyActivity } from "./LearningSession";
import UiIcon from "./UiIcon";
import ActionMenu from "./ActionMenu";
import { getPdfContext, getPdfFigures, isPdfFile, likelyStudyImage, prepareNoteImage, searchStudyImages } from "./documentMedia";
import { markdownToNoteHtml, noteHtmlToMarkdown, sanitizeNoteHtml } from "./noteFormat";
import { downloadStudyFile, sendStudyRequest } from "./studyTools";
import "./documentStudio.css";
import "./workspace.css";

const editorExtensions = [
  NoteEquation,
  StarterKit.configure({ link: { openOnClick: false, HTMLAttributes: { rel: "noopener noreferrer" } } }),
  Image.configure({ allowBase64: true, resize: { enabled: true, minWidth: 60, minHeight: 45, alwaysPreserveAspectRatio: true }, HTMLAttributes: { referrerpolicy: "no-referrer" } }),
  TableKit.configure({ table: { resizable: true, renderWrapper: true } }),
  TextStyleKit.configure({ color: false, backgroundColor: false, lineHeight: false }),
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

export default function NotesEditor({ file, fileObj, getFileText, onUpdate, readSavedNotes, initialMode, onModeChange }) {
  const [error, setError] = useState("");
  const [equation, setEquation] = useState("E = mc^2");
  const [savedAt, setSavedAt] = useState(0);
  const [status, setStatus] = useState("");
  const [topic, setTopic] = useState("");
  const [style, setStyle] = useState("Detailed");
  const [language, setLanguage] = useState("Same as source");
  const [webImages, setWebImages] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [panel, setPanel] = useState(file.notesMode === "generate" ? "generate" : null);
  const [view, setView] = useState(initialMode || ((file.notesMode || (file.defaultTab === "learn" ? "lessons" : "write")) === "lessons" ? "lessons" : "write"));
  useEffect(() => { if (initialMode) setView(initialMode); }, [initialMode]);
  useStudyActivity(view === "write" ? file.id : null, "Notes", file.notes || "", file.name);
  const workspace = useRef(null);
  const [moreTools, setMoreTools] = useState(false);
  const [narrow, setNarrow] = useState(() => window.innerWidth < 900);
  const panelRef = useRef(null);
  const latest = useRef({ file, onUpdate });
  latest.current = { file, onUpdate };
  const updateFields = patch => {
    const next = { ...latest.current.file, ...patch };
    latest.current.file = next;
    latest.current.onUpdate(next);
  };
  const [imageQuery, setImageQuery] = useState("");
  const [pdfPage, setPdfPage] = useState(1);
  const [searching, setSearching] = useState(false);
  const [images, setImages] = useState([]);
  const [saved, setSaved] = useState(file.namedNotes || []);
  const [versionName, setVersionName] = useState("");
  const [preview, setPreview] = useState(false);
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
      updateFields({ notesHtml, notes: noteHtmlToMarkdown(notesHtml) });
      setSavedAt(Date.now());
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
      fontFamily: current?.getAttributes("textStyle").fontFamily,
      fontSize: current?.getAttributes("textStyle").fontSize,
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

  useEffect(() => {
    if (!workspace.current) return;
    const node = workspace.current, update = () => setNarrow(node.clientWidth < 800);
    update();
    const observer = new ResizeObserver(update); observer.observe(node);
    return () => observer.disconnect();
  }, [editor]);

  useEffect(() => {
    if (!narrow || !panel || !panelRef.current) return;
    const before = document.activeElement, overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panelRef.current.querySelector("button")?.focus();
    const keyboard = event => {
      if (event.key === "Escape") setPanel(null);
      if (event.key !== "Tab") return;
      const controls = [...panelRef.current.querySelectorAll("button:not(:disabled), input:not([hidden]), select, summary, a[href]")];
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    };
    window.addEventListener("keydown", keyboard);
    return () => { document.body.style.overflow = overflow; window.removeEventListener("keydown", keyboard); if (before?.isConnected) before.focus(); };
  }, [narrow, panel, editor]);

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
        : pdfFile ? await getPdfContext(pdfFile) : (await getFileText?.()) || existing;
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
    updateFields({ notes, notesHtml: html, namedNotes: next });
    setStatus(`Saved version: ${name}.`);
  }

  if (!editor) return <section className="note-initial-frame" aria-busy="true" aria-label="Notes" />;
  const run = operation => operation(editor.chain().focus()).run();
  const action = (label, icon, operation, pressed) => <button type="button" title={label} aria-label={label}
    disabled={preview} aria-pressed={pressed} onMouseDown={event => event.preventDefault()} onClick={() => run(operation)}><UiIcon name={icon} /></button>;
  const openPanel = name => { setView("write"); onModeChange?.("write"); setPanel(value => value === name ? null : name); };
  const exportNotes = () => {
    const content = document.createElement("div"); content.innerHTML = sanitizeNoteHtml(editor.getHTML());
    content.querySelectorAll("[data-classio-equation]").forEach(node => { node.innerHTML = katex.renderToString(node.getAttribute("data-latex") || "", { output: "mathml", throwOnError: false, trust: false, maxExpand: 500 }); });
    downloadStudyFile(new Blob([`<!doctype html><meta charset="utf-8"><title>Classio notes</title><style>body{max-width:760px;margin:40px auto;padding:20px;font:16px/1.7 system-ui;color:#253648}table{border-collapse:collapse;width:100%}td,th{border:1px solid #c7d5e0;padding:.5rem}th{background:#edf4fa}img{max-width:100%}math{font-size:1.15em}</style>${content.innerHTML}`], { type: "text/html" }), "classio-notes.html");
  };

  const applyDraft = append => {
    const chain = editor.chain().focus(append ? "end" : undefined).command(({ tr }) => { closeHistory(tr); return true; });
    if (!append) chain.selectAll();
    chain.insertContent(draft).run();
    setDraft(""); setPanel(null); updateFields({ notesMode: "write" });
    setStatus("Notes updated. You can undo this change.");
  };

  return <section ref={workspace} className="notes-studio notes-workspace" aria-label="Notes workspace">
    <header className="notes-workspace-header">
      <nav className="workspace-tabs" aria-label="Notes modes">
        <button type="button" aria-pressed={view === "write"} onClick={() => { setView("write"); onModeChange?.("write"); updateFields({ notesMode: "write" }); }}>Notes</button>
        <button type="button" aria-pressed={view === "lessons"} onClick={() => { setView("lessons"); onModeChange?.("lessons"); setPanel(null); updateFields({ notesMode: "lessons" }); }}>Lessons</button>
      </nav>
      <div className="notes-primary-actions">
        <button className="quiet-button is-primary" type="button" aria-expanded={panel === "generate"} onClick={() => openPanel("generate")}><UiIcon name="generate" /><span>Generate</span></button>
        <ActionMenu label="Note options" items={[
          { label: "Download notes", icon: "download", run: exportNotes },
          ...(file.transcript ? [{ label: "Download transcript", run: () => downloadStudyFile(new Blob([file.transcript], { type: "text/plain" }), "classio-transcript.txt") }] : []),
          { label: "Saved versions", run: () => openPanel("versions") },
          { label: preview ? "Edit notes" : "Reading view", run: () => { editor.setEditable(preview); setPreview(value => !value); } },
        ]} />
      </div>
    </header>
    {view === "lessons" ? <LearnMode file={file} getFileText={getFileText} onUpdate={next => updateFields({ course: next.course })}
      onUseAsNotes={markdown => {
        editor.chain().focus("end").command(({ tr }) => { closeHistory(tr); return true; }).insertContent(markdownToNoteHtml(markdown)).run();
        setView("write"); onModeChange?.("write"); updateFields({ notesMode: "write" }); setStatus("Lessons added to your notes.");
      }} /> : <div className={`notes-layout ${panel && !narrow ? "has-inspector" : ""}`}>
      <article className="note-workspace">
        <div className="note-toolbar" role="toolbar" aria-label="Note formatting">
          <select aria-label="Font family" disabled={preview} value={tools?.fontFamily || "DM Sans Variable"} onChange={event => run(chain => chain.setFontFamily(event.target.value))}>
            <option value="DM Sans Variable">Sans</option><option value="Fraunces Variable">Serif</option><option value="ui-monospace">Mono</option>
          </select>
          <select aria-label="Font size" disabled={preview} value={tools?.fontSize || "16px"} onChange={event => run(chain => chain.setFontSize(event.target.value))}>
            {[12, 14, 16, 18, 20, 24, 28, 32].map(size => <option key={size} value={`${size}px`}>{size}</option>)}
          </select>
          <span className="toolbar-divider" />
          {action("Bold", "bold", chain => chain.toggleBold(), tools?.bold)}
          {action("Italic", "italic", chain => chain.toggleItalic(), tools?.italic)}
          {action("Underline", "underline", chain => chain.toggleUnderline(), tools?.underline)}
          <button type="button" disabled={preview} aria-label="Insert equation" title="Insert equation" onClick={() => openPanel("equation")}><UiIcon name="equation" /></button>
          {action("Bullets", "list", chain => chain.toggleBulletList(), tools?.bullet)}
          <span className="toolbar-divider" />
          <button type="button" disabled={preview} aria-label="Insert table" title="Insert table" onMouseDown={event => event.preventDefault()} onClick={() => run(chain => chain.insertTable({ rows: 3, cols: 3, withHeaderRow: true }))}><UiIcon name="table" /></button>
          <button type="button" disabled={preview} aria-label="Insert image" title="Insert image" aria-expanded={panel === "images"} onClick={() => openPanel("images")}><UiIcon name="image" /></button>
          <button type="button" aria-label="More formatting" title="More formatting" aria-expanded={moreTools} onClick={() => setMoreTools(value => !value)}><UiIcon name="more" /></button>
        </div>
        {moreTools && <div className="studio-controls note-more-formatting" aria-label="Additional formatting">
          <select aria-label="Paragraph style" disabled={preview} value={tools?.heading || 0} onChange={event => {
            const level = Number(event.target.value); run(chain => level ? chain.setHeading({ level }) : chain.setParagraph());
          }}><option value="0">Paragraph</option><option value="1">Title</option><option value="2">Heading</option><option value="3">Subheading</option></select>
          <button type="button" disabled={preview} aria-pressed={tools?.ordered} onClick={() => run(chain => chain.toggleOrderedList())}>Numbered list</button>
          <button type="button" disabled={preview} aria-pressed={tools?.callout} onClick={() => run(chain => chain.toggleBlockquote())}>Callout</button>
          <button type="button" disabled={preview} aria-pressed={tools?.strike} onClick={() => run(chain => chain.toggleStrike())}>Strike</button>
          <select aria-label="Align text" disabled={preview} value="" onChange={event => run(chain => chain.setTextAlign(event.target.value))}>
            <option value="" disabled>Align</option><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option><option value="justify">Justify</option>
          </select>
          {action("Undo", "undo", chain => chain.undo())}{action("Redo", "redo", chain => chain.redo())}
        </div>}
        {tools?.table && <div className="studio-controls note-table-tools" aria-label="Table actions">
          <button onClick={() => run(chain => chain.addRowAfter())}>Add row</button><button onClick={() => run(chain => chain.addColumnAfter())}>Add column</button>
          <button onClick={() => run(chain => chain.deleteRow())}>Delete row</button><button onClick={() => run(chain => chain.deleteColumn())}>Delete column</button>
          <button onClick={() => run(chain => chain.mergeCells())}>Merge cells</button><button onClick={() => run(chain => chain.splitCell())}>Split cell</button>
          <button onClick={() => run(chain => chain.deleteTable())}>Delete table</button>
        </div>}
        {tools?.image && !preview && <div className="studio-controls note-table-tools"><button onClick={() => run(chain => chain.deleteSelection())}>Remove selected image</button></div>}
        <div className="note-document-meta"><span>{editor.getText().trim() ? editor.getText().trim().split(/\s+/).length : 0} words</span><span>{savedAt ? "Saved in workspace" : "Changes save automatically"}</span></div>
        <div className={`note-paper ${preview ? "is-reading" : ""}`}><EditorContent editor={editor} /></div>
        {status && !busy && <p className="note-local-status" role="status">{status}</p>}
      </article>
      {panel && <>
        {narrow && <button className="notes-panel-backdrop" aria-label="Close note tools backdrop" onClick={() => setPanel(null)} />}
        <aside ref={panelRef} className={`notes-inspector ${narrow ? "is-drawer" : ""}`} role={narrow ? "dialog" : "complementary"} aria-modal={narrow || undefined}
          aria-label={panel === "generate" ? "AI note tools" : panel === "images" ? "Note images" : panel === "equation" ? "Equation editor" : "Saved versions"}>
          <header><h2>{panel === "generate" ? "Generate notes" : panel === "images" ? "Images" : panel === "equation" ? "Equation" : "Versions"}</h2>
            <button type="button" className="quiet-button" aria-label="Close note tools" onClick={() => setPanel(null)}><UiIcon name="close" /></button></header>
          {panel === "generate" && <div className="notes-panel-body">
            <p>Start with a topic or use the material in this file.</p>
            <label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with source material" /></label>
            <div className="notes-panel-options"><label>Style<select aria-label="Note style" value={style} onChange={event => setStyle(event.target.value)}><option>Detailed</option><option>Exam review</option><option>Concise summary</option><option>Simple explanations</option></select></label>
              <label>Language<select aria-label="Note language" value={language} onChange={event => setLanguage(event.target.value)}><option>Same as source</option><option>English</option><option>Arabic</option><option>Chinese</option><option>French</option><option>Spanish</option></select></label></div>
            <label className="studio-check"><input type="checkbox" checked={webImages} onChange={event => setWebImages(event.target.checked)} /> Include relevant images</label>
            <button className="quiet-button is-primary" type="button" disabled={busy} onClick={() => generate("generate")}>{busy ? "Generating…" : "Generate notes"}</button>
            <details><summary>Use my writing</summary><div className="studio-controls">
              <button type="button" disabled={busy} onClick={() => generate("organize")}>Organize notes</button>
              <button type="button" disabled={busy} onClick={() => generate("table")}>Make a table</button></div></details>
            {error && <p className="studio-error" role="alert">{error}</p>}
            {draft && <section className="note-draft" aria-label="AI draft"><h3>Review draft</h3>
              <div className="classio-note-content" dangerouslySetInnerHTML={{ __html: draft }} />
              <div className="studio-controls"><button type="button" onClick={() => applyDraft(false)}>Replace notes</button><button type="button" onClick={() => applyDraft(true)}>Append</button><button type="button" onClick={() => setDraft("")}>Discard</button></div>
            </section>}
          </div>}
          {panel === "images" && <div className="notes-panel-body">
            <button type="button" className="quiet-button" onClick={() => upload.current?.click()}><UiIcon name="image" />Upload image</button>
            <input ref={upload} type="file" accept="image/png,image/jpeg,image/webp,image/gif" hidden onChange={event => { const image = event.target.files[0]; event.target.value = ""; if (image) uploadImage(image); }} />
            {pdfFile && <form onSubmit={event => { event.preventDefault(); findImages("pdf"); }}><label>PDF page<input type="number" min="1" value={pdfPage} onChange={event => setPdfPage(event.target.value)} /></label><button type="submit" className="quiet-button" disabled={searching}>Extract figures</button></form>}
            <form onSubmit={event => { event.preventDefault(); findImages("web"); }}><label>Search images<input value={imageQuery} onChange={event => setImageQuery(event.target.value)} placeholder="e.g. plant cell diagram" /></label>
              <button type="submit" className="quiet-button" disabled={searching || !imageQuery.trim()}>Search Wikimedia</button></form>
            {error && <p role="alert" className="studio-error">{error}</p>}
            <div className="note-image-results">{images.map(image => <figure key={image.src}><img src={image.src} alt={image.title} referrerPolicy="no-referrer" loading="lazy" />
              <figcaption>{image.source ? <a href={image.source} target="_blank" rel="noreferrer">{image.credit}</a> : image.title}</figcaption><button onClick={() => addFoundImage(image)}>Insert image</button></figure>)}</div>
          </div>}
          {panel === "equation" && <form className="notes-panel-body" onSubmit={event => { event.preventDefault(); if (!equation.trim()) return; editor.chain().focus().insertContent({ type: "noteEquation", attrs: { latex: equation.trim().slice(0, 600) } }).run(); setPanel(null); }}>
            <label>Equation (LaTeX)<input aria-label="Equation LaTeX" maxLength={600} value={equation} onChange={event => setEquation(event.target.value)} /></label><div className="equation-preview" aria-label="Equation preview" dangerouslySetInnerHTML={{ __html: katex.renderToString(equation, { throwOnError: false, trust: false, maxExpand: 500 }) }} /><button className="quiet-button is-primary" disabled={!equation.trim()}>Insert equation</button><p>Use LaTeX notation, for example \(\frac&#123;a&#125;&#123;b&#125;\), x^2, or \sqrt&#123;x&#125;.</p>
          </form>}
          {panel === "versions" && <div className="notes-panel-body"><form onSubmit={saveVersion}><label>Version name<input value={versionName} onChange={event => setVersionName(event.target.value)} placeholder="e.g. Before exam review" /></label>
            <button className="quiet-button" type="submit" disabled={!versionName.trim()}>Save version</button></form>
            {saved.length > 0 && <label>Restore version<select aria-label="Restore version" value="" onChange={event => { const version = saved[Number(event.target.value)]; if (version) editor.commands.setContent(version.html ? sanitizeNoteHtml(version.html) : markdownToNoteHtml(version.text)); }}>
              <option value="" disabled>Choose a version</option>{saved.map((version, index) => <option key={version.name} value={index}>{version.name}</option>)}</select></label>}
          </div>}
        </aside>
      </>}
    </div>}
  </section>;
}
