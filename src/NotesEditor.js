import ChoiceMenu from "./ChoiceMenu";
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
import StudioDialog from "./StudioDialog";
import { MATH_OUTPUT_GUIDE, typesetMath } from "./mathMarkdown";
import { enrichNoteImages, insertNoteFigure } from "./noteFigures";
import { getPdfContext, getPdfFigures, isPdfFile, prepareNoteImage, searchStudyImages } from "./documentMedia";
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
  const [panel, setPanel] = useState(file.notesMode === "generate" ? "generate" : null);
  const [view, setView] = useState(initialMode || ((file.notesMode || (file.defaultTab === "learn" ? "lessons" : "write")) === "lessons" ? "lessons" : "write"));
  useEffect(() => { if (initialMode) setView(initialMode); }, [initialMode]);
  useStudyActivity(view === "write" ? file.id : null, "Notes", file.notes || "", file.name);
  const workspace = useRef(null);
  const [moreTools, setMoreTools] = useState(false);
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
    insertNoteFigure(imageHtml, image, image.title, marker); marker.remove();
    editor.chain().focus().insertContent(sanitizeNoteHtml(imageHtml.innerHTML)).run();
    setStatus(`Added ${image.title}.`);
  }

  async function placeSuggestedImages() {
    if (searching || busy) return;
    mediaRequest.current?.abort(); const controller = new AbortController(); mediaRequest.current = controller;
    const originalHtml = editor.getHTML(); setSearching(true); setError(""); setStatus("Finding illustrations for your notes…");
    try {
      const result = await enrichNoteImages("", { existingHtml: originalHtml, pdfFile, fileObj, useWeb: true, signal: controller.signal });
      if (controller.signal.aborted) return;
      if (editor.getHTML() !== originalHtml) { setStatus("Your notes changed during the image search. Choose Place suggested images again when you’re ready."); return; }
      editor.chain().command(({ tr }) => { closeHistory(tr); return true; }).selectAll().insertContent(result.html).run();
      setPanel(current => current === "images" ? null : current); setStatus((result.added ? `${result.added} illustration${result.added === 1 ? "" : "s"} placed beside the related explanation.` : "Image suggestions cleaned up.") + (result.missing ? " Some images were unavailable; you can choose another image with Images." : "") + " You can undo this change.");
    } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
    finally { if (!controller.signal.aborted) setSearching(false); }
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
        { role: "system", content: `${instruction} Language: ${language}. ${MATH_OUTPUT_GUIDE} Return Markdown only. Do not write raw HTML or external URLs. ${pdfFile ? "For up to three relevant source figures, put [[PDF:3]] on a separate line with an actual page number from the source markers." : ""} ${webImages && action !== "table" ? "Include up to three relevant illustrations using [[IMAGE:specific subject]] on their own lines immediately after the paragraph that explains that subject. Never collect image searches or figure markers into a section at the end. Skip decorative or uncertain images." : ""}` },
        { role: "user", content: `Topic: ${topic.trim() || file.name}\n\n${(source || "").slice(0, 16000)}` },
      ], maxTokens: action === "table" ? 1500 : 5500 }, controller.signal);
      const enriched = await enrichNoteImages(raw.replace(/^\s*```(?:markdown|md)?\s*\n|\n```\s*$/g, ""), { pdfFile, fileObj, useWeb: webImages, signal: controller.signal, originalHtml: action === "organize" ? originalHtml : "" });
      const html = enriched.html;
      if (!controller.signal.aborted) {
        if (action === "table") {
          const document = new DOMParser().parseFromString(html, "text/html");
          const table = document.querySelector("table");
          if (!table) throw new Error("The AI did not return a table. Try a clearer set of notes.");
          editor.chain().focus().insertContent(table.outerHTML).run();
          setStatus("Table inserted. You can edit every cell.");
        } else {
          const wasEmpty = editor.isEmpty;
          const replace = action === "organize" && editor.getHTML() === originalHtml;
          const chain = editor.chain().command(({ tr }) => { closeHistory(tr); return true; });
          if (replace) chain.selectAll(); else chain.focus("end");
          chain.insertContent(html).run();
          editor.setEditable(true); setPreview(false); setView("write"); onModeChange?.("write");
          updateFields({ notesMode: "write" }); setPanel(null);
          setStatus((wasEmpty ? "Notes generated and saved in your notebook." : replace ? "Notes organized. Undo restores your previous writing." : "Generated notes added below your writing. You can undo this change.") + (enriched.missing ? " Some illustrations were unavailable. Use Images to choose another source." : ""));
          editor.commands.focus(wasEmpty ? "start" : "end", { scrollIntoView: false });
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
    typesetMath(content, "mathml");
    downloadStudyFile(new Blob([`<!doctype html><meta charset="utf-8"><title>Classio notes</title><style>body{max-width:760px;margin:40px auto;padding:20px;font:16px/1.7 system-ui;color:#253648}table{border-collapse:collapse;width:100%}td,th{border:1px solid #c7d5e0;padding:.5rem}th{background:#edf4fa}img{max-width:100%}math{font-size:1.15em}</style>${content.innerHTML}`], { type: "text/html" }), "classio-notes.html");
  };

  return <section ref={workspace} className="notes-studio notes-workspace" aria-label="Notes workspace">
    <header className="notes-workspace-header">
      <div className="notes-mode-choice"><h2>Notes</h2><ChoiceMenu aria-label="Notes mode" variant="notebook" icon={view === "lessons" ? "book" : "notes"} descriptions={{ write: "Write freely or generate notes into your notebook.", lessons: "Learn step by step, with worked examples and practice questions." }} value={view} onChange={event => { const mode = event.target.value; setView(mode); setPanel(null); onModeChange?.(mode); updateFields({ notesMode: mode }); }}><option value="write">Write & generate notes</option><option value="lessons">Guided lessons with questions</option></ChoiceMenu></div>
      <div className="notes-primary-actions">
        <button className="quiet-button is-primary" type="button" hidden={view === "lessons"} aria-expanded={panel === "generate"} onClick={() => openPanel("generate")}><UiIcon name="generate" /><span>Generate</span></button>
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
      }} /> : <div className="notes-layout">
      <article className="note-workspace">
        <div className="note-toolbar" role="toolbar" aria-label="Note formatting">
          <ChoiceMenu aria-label="Font family" disabled={preview} value={tools?.fontFamily || "DM Sans Variable"} onChange={event => run(chain => chain.setFontFamily(event.target.value))}>
            <option value="DM Sans Variable">Sans</option><option value="Fraunces Variable">Serif</option><option value="ui-monospace">Mono</option>
          </ChoiceMenu>
          <ChoiceMenu aria-label="Font size" disabled={preview} value={tools?.fontSize || "16px"} onChange={event => run(chain => chain.setFontSize(event.target.value))}>
            {[12, 14, 16, 18, 20, 24, 28, 32].map(size => <option key={size} value={`${size}px`}>{size}</option>)}
          </ChoiceMenu>
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
          <ChoiceMenu aria-label="Paragraph style" disabled={preview} value={tools?.heading || 0} onChange={event => {
            const level = Number(event.target.value); run(chain => level ? chain.setHeading({ level }) : chain.setParagraph());
          }}><option value="0">Paragraph</option><option value="1">Title</option><option value="2">Heading</option><option value="3">Subheading</option></ChoiceMenu>
          <button type="button" disabled={preview} aria-pressed={tools?.ordered} onClick={() => run(chain => chain.toggleOrderedList())}>Numbered list</button>
          <button type="button" disabled={preview} aria-pressed={tools?.callout} onClick={() => run(chain => chain.toggleBlockquote())}>Callout</button>
          <button type="button" disabled={preview} aria-pressed={tools?.strike} onClick={() => run(chain => chain.toggleStrike())}>Strike</button>
          <ChoiceMenu aria-label="Align text" disabled={preview} value="" onChange={event => run(chain => chain.setTextAlign(event.target.value))}>
            <option value="" disabled>Align</option><option value="left">Left</option><option value="center">Center</option><option value="right">Right</option><option value="justify">Justify</option>
          </ChoiceMenu>
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
        {busy && <div className="note-generation-status" role="status"><span className="mini-spinner" />Generating notes for your notebook…<button className="text-button" onClick={() => { aiRequest.current?.abort(); aiRequest.current = null; setBusy(false); setStatus("Generation cancelled. Your writing is unchanged."); }}>Cancel generation</button></div>}
        <div className={`note-paper ${preview ? "is-reading" : ""}`}><EditorContent editor={editor} /></div>
        {status && !busy && <p className="note-local-status" role="status">{status}</p>}
      </article>
      {panel && <StudioDialog title={panel === "generate" ? "Generate notes" : panel === "images" ? "Images" : panel === "equation" ? "Equation" : "Saved versions"} onClose={() => setPanel(null)}>
        <div className="notes-tool-dialog">
          {panel === "generate" && <div className="notes-panel-body">
            <p>Notes go straight into your notebook. Existing writing is kept; you can undo generation.</p>
            <label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with source material" /></label>
            <div className="notes-panel-options"><label>Style<ChoiceMenu aria-label="Note style" value={style} onChange={event => setStyle(event.target.value)}><option>Detailed</option><option>Exam review</option><option>Concise summary</option><option>Simple explanations</option></ChoiceMenu></label>
              <label>Language<ChoiceMenu aria-label="Note language" value={language} onChange={event => setLanguage(event.target.value)}><option>Same as source</option><option>English</option><option>Arabic</option><option>Chinese</option><option>French</option><option>Spanish</option></ChoiceMenu></label></div>
            <label className="studio-check"><input type="checkbox" checked={webImages} onChange={event => setWebImages(event.target.checked)} /> Include relevant images</label>
            <button className="quiet-button is-primary" type="button" disabled={busy} onClick={() => generate("generate")}>{busy ? "Generating…" : "Generate notes"}</button>
            <details><summary>Use my writing</summary><div className="studio-controls">
              <button type="button" disabled={busy} onClick={() => generate("organize")}>Organize notes</button>
              <button type="button" disabled={busy} onClick={() => generate("table")}>Make a table</button></div></details>
            {error && <p className="studio-error" role="alert">{error}</p>}

          </div>}
          {panel === "images" && <div className="notes-panel-body">
            {/\[\[(?:IMAGE|PDF):/i.test(editor.getText()) && <><p>Your notes include suggested illustrations. Place real images next to their related explanations while keeping your writing and formatting.</p><button type="button" className="quiet-button is-primary" disabled={searching || busy} onClick={placeSuggestedImages}>{searching ? "Placing images…" : "Place suggested images"}</button></>}
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
            {saved.length > 0 && <label>Restore version<ChoiceMenu aria-label="Restore version" value="" onChange={event => { const version = saved[Number(event.target.value)]; if (version) editor.commands.setContent(version.html ? sanitizeNoteHtml(version.html) : markdownToNoteHtml(version.text)); }}>
              <option value="" disabled>Choose a version</option>{saved.map((version, index) => <option key={version.name} value={index}>{version.name}</option>)}</ChoiceMenu></label>}
          </div>}
        </div>
      </StudioDialog>}
    </div>}
  </section>;
}
