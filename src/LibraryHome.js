import { useMemo, useState } from "react";
import ActionMenu from "./ActionMenu";
import UiIcon from "./UiIcon";
import WorkspaceArt from "./WorkspaceArt";
import InlineAssistant from "./InlineAssistant";

export function materialKind(file) {
  const name = (file.name || "").toLowerCase(), type = file.type || "";
  if (type === "text/youtube" || file.youtubeUrl) return { label: "Link", icon: "link", key: "links" };
  if (/pdf$/.test(name) || type.includes("pdf")) return { label: "PDF", icon: "source", key: "documents" };
  if (/pptx?$/.test(name) || type.includes("presentation")) return { label: "Slides", icon: "slides", key: "documents" };
  if (type.startsWith("audio/")) return { label: "Audio", icon: "mic", key: "audio" };
  if (file._isBlank || file.isBlankDoc || file.id?.startsWith("blank_")) return { label: file.course?.units?.length ? "Course & notes" : "Note", icon: "notes", key: "notes" };
  return { label: "Document", icon: "source", key: "documents" };
}
const shortPreview = file => (file.notes || file.transcript || "").replace(/<[^>]*>/g, " ").replace(/!?\[([^\]]*)\]\([^)]+\)/g, "$1").replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&").replace(/[#>*_`|]/g, "").replace(/\s+/g, " ").trim().slice(0, 125);

export default function LibraryHome({ folders, guest, signIn, signInPending, signInError, createNote, upload, record, addLink,
  newFolder, openFile, openFolder, rename, remove, move, onAsk = () => {} }) {
  const [query, setQuery] = useState(""), [filter, setFilter] = useState("all"), [sort, setSort] = useState("recent");
  const [layout, setLayout] = useState("grid");
  const all = useMemo(() => folders.flatMap(folder => (folder.files || []).map(file => ({ file, folder }))), [folders]);
  const subjects = folders.filter(folder => folder.id !== "inbox");
  const matchedFolders = subjects.filter(folder => folder.name.toLowerCase().includes(query.toLowerCase()));
  const items = all.filter(({ file, folder }) => (!query || `${file.name} ${folder.name} ${file.notes || ""}`.toLowerCase().includes(query.toLowerCase())) && (filter === "all" || materialKind(file).key === filter))
    .sort((a, b) => sort === "name" ? a.file.name.localeCompare(b.file.name) : sort === "type" ? materialKind(a.file).label.localeCompare(materialKind(b.file).label) || a.file.name.localeCompare(b.file.name)
      : (b.file.lastOpenedAt || b.file.createdAt || 0) - (a.file.lastOpenedAt || a.file.createdAt || 0));
  const recent = [...all].filter(item => item.file.lastOpenedAt).sort((a, b) => b.file.lastOpenedAt - a.file.lastOpenedAt).slice(0, 3);
  const actions = [
    { label: "Blank note", icon: "notes", run: () => createNote("write") }, { label: "Generate notes", icon: "generate", run: () => createNote("generate") },
    { label: "Build a lesson", icon: "quiz", run: () => createNote("lessons") }, { separator: true },
    { label: "Upload document", icon: "source", run: upload }, { label: "Record or upload audio", icon: "mic", run: record },
    { label: "Add a link", icon: "link", run: addLink }, { separator: true }, { label: "Folder", icon: "folder", run: newFolder },
  ];
  const fileActions = (file, folder) => [
    { label: "Rename", run: () => rename({ ...file, type: "file", scope: folder.id }) },
    ...(folders.some(item => item.id !== folder.id) ? [{ label: "Move to", items: folders.filter(item => item.id !== folder.id).map(item => ({ label: item.name, run: () => move(file, folder.id, item.id) })) }] : []),
    { separator: true }, { label: "Delete", danger: true, run: () => remove({ ...file, type: "file" }) },
  ];
  return <section className="library-home premium-library" aria-label="Your library">
    <header className="library-heading"><div><p className="eyebrow">YOUR STUDY SPACE</p><h1>Library</h1><p>Bring your material. Choose how to study it.</p></div>
      <ActionMenu label="Create new" primary icon="plus" items={actions}>Create new</ActionMenu></header>
    {guest && <div className="library-guest"><p><strong>You’re studying as a guest.</strong> Every tool is free. Sign in to save and sync; guest work clears when the session ends.</p>
      <button className="text-button" disabled={signInPending} onClick={signIn}>{signInPending ? "Connecting…" : "Sign in to save"}</button>{signInError && <p role="alert">{signInError}</p>}</div>}
    <section className="library-quick-actions" aria-label="Start studying">
      {[
        { label: "Blank document", description: "Write and organize your notes", icon: "notes", run: () => createNote("write"), tone: "blue" },
        { label: "Upload document", description: "PDF, Word or PowerPoint", icon: "source", run: upload, tone: "sand" },
        { label: "Record audio", description: "Record or transcribe a lecture", icon: "mic", run: record, tone: "rose" },
        { label: "Website link", description: "YouTube transcripts and webpages", icon: "link", run: addLink, tone: "blue" },
      ].map(item => <button type="button" className={`creation-card tone-${item.tone}`} aria-label={item.label} title={item.description} key={item.label} onClick={item.run}><span className="creation-icon"><UiIcon name={item.icon} size={23} /></span>
        <strong>{item.label}</strong><span>{item.description}</span><UiIcon name="arrow" size={17} /></button>)}
    </section>
    <InlineAssistant onAsk={onAsk} contextLabel="your library" />
    {!!recent.length && !query && filter === "all" && <section className="library-recent" aria-label="Recently opened"><div className="section-heading"><h2>Pick up where you left off</h2><span>Recently opened</span></div>
      <div className="recent-grid">{recent.map(({ file, folder }) => <button key={file.id} onClick={() => openFile(file, folder)}><UiIcon name={materialKind(file).icon} /><span><strong>{file.name}</strong><small>{folder.id === "inbox" ? "Library" : folder.name}</small></span><UiIcon name="arrow" size={16} /></button>)}</div></section>}
    <section className="library-materials" aria-label="Learning materials"><div className="section-heading"><h2>Your materials</h2><button className="text-button" onClick={newFolder}><UiIcon name="plus" size={15} />New folder</button></div>
      <div className="library-filters"><label className="library-search"><UiIcon name="search" /><input type="search" aria-label="Search library" placeholder="Search notes, files or folders" value={query} onChange={event => setQuery(event.target.value)} /></label>
        <select aria-label="Filter materials" value={filter} onChange={event => setFilter(event.target.value)}><option value="all">All materials</option><option value="notes">Notes & courses</option><option value="documents">Documents</option><option value="audio">Audio</option><option value="links">Links</option></select>
        <select aria-label="Sort materials" value={sort} onChange={event => setSort(event.target.value)}><option value="recent">Recently opened</option><option value="name">Name A–Z</option><option value="type">File type</option></select>
        <button className="quiet-button library-layout-toggle" aria-label={layout === "grid" ? "Switch to list view" : "Switch to card view"} onClick={() => setLayout(value => value === "grid" ? "list" : "grid")}><UiIcon name={layout === "grid" ? "list" : "grid"} /></button></div>
      {!!matchedFolders.length && filter === "all" && <div className="library-folder-grid" aria-label="Folders">{matchedFolders.map(folder => <article className="folder-card" key={folder.id}>
        <button className="folder-open" aria-label={`Open folder ${folder.name}`} onClick={() => openFolder(folder)}><span style={{ color: folder.color || "var(--app-link)" }}><UiIcon name="folder" size={28} /></span><strong>{folder.name}</strong><small>{folder.files?.length || 0} {(folder.files?.length || 0) === 1 ? "file" : "files"}</small></button>
        <ActionMenu label={`Actions for ${folder.name}`} items={[{ label: "Rename", run: () => rename({ ...folder, type: "folder" }) }, { label: "Delete", danger: true, run: () => remove({ ...folder, type: "folder" }) }]} /></article>)}</div>}
      {items.length ? <div className={`material-grid is-${layout}`} aria-label="Notes and files">{items.map(({ file, folder }) => { const kind = materialKind(file); return <article className="material-card" key={file.id}>
        <button className="material-open" aria-label={`Open ${file.name}`} onClick={() => openFile(file, folder)}><div className={`material-preview type-${kind.key}`}><UiIcon name={kind.icon} size={32} /><span>{kind.label}</span>{shortPreview(file) && <p>{shortPreview(file)}</p>}</div>
          <div className="material-info"><h3>{file.name}</h3><p>{folder.id === "inbox" ? "Library" : folder.name}{file.course?.units?.length ? ` · ${file.course.units.length} lessons` : file.studyCards?.length ? ` · ${file.studyCards.length} cards` : ""}</p></div></button>
        <ActionMenu label={`Actions for ${file.name}`} items={fileActions(file, folder)} /></article>; })}</div>
        : <div className="library-empty"><WorkspaceArt /><h2>{query || filter !== "all" ? "No matching materials" : "Start with something you’re learning"}</h2>
          <p>{query || filter !== "all" ? "Try a different search or filter. Your other materials are still in the Library." : "Add a document, record a lecture, or write your first note. Then create lessons, quizzes and flashcards from it."}</p>
          {query || filter !== "all" ? <button className="quiet-button" onClick={() => { setQuery(""); setFilter("all"); }}>Clear filters</button> : <button className="quiet-button is-primary" onClick={upload}><UiIcon name="plus" />Add your first document</button>}</div>}
    </section>
  </section>;
}
