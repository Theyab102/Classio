import ActionMenu from "./ActionMenu";
import UiIcon from "./UiIcon";

export default function LibraryHome({ folders, guest, signIn, signInPending, signInError, createNote, upload, record, addLink,
  newFolder, openFile, openFolder, rename, remove, move }) {
  const inbox = folders.find(folder => folder.id === "inbox");
  const subjects = folders.filter(folder => folder.id !== "inbox");
  const newActions = [
    { label: "Blank note", icon: "notes", run: () => createNote("write") },
    { label: "Generate notes", icon: "generate", run: () => createNote("generate") },
    { label: "Build a lesson", icon: "quiz", run: () => createNote("lessons") },
    { separator: true },
    { label: "Upload document", icon: "source", run: upload },
    { label: "Record or upload audio", icon: "mic", run: record },
    { label: "Add a link", icon: "link", run: addLink },
    { separator: true }, { label: "Folder", icon: "folder", run: newFolder },
  ];
  const fileItems = (file, folder) => [
    { label: "Rename", run: () => rename({ ...file, type: "file", scope: folder.id }) },
    ...(folders.some(item => item.id !== folder.id) ? [{ label: "Move to", items: folders.filter(item => item.id !== folder.id)
      .map(item => ({ label: item.name, run: () => move(file, folder.id, item.id) })) }] : []),
    { separator: true }, { label: "Delete", danger: true, run: () => remove({ ...file, type: "file" }) },
  ];
  return <section className="library-home" aria-label="Your library">
    <header className="library-heading"><div><h1>Library</h1><p>Your notes and study files.</p></div>
      <ActionMenu label="Create new" primary icon="plus" items={newActions}>New</ActionMenu>
    </header>
    {guest && <div className="library-guest"><p><strong>Guest workspace.</strong> Sign in to save and sync. Guest work clears when the session ends.</p>
      <button type="button" className="text-button" disabled={signInPending} onClick={signIn}>{signInPending ? "Connecting…" : "Sign in to save"}</button>
      {signInError && <p role="alert">{signInError}</p>}
    </div>}
    {!subjects.length && !inbox?.files?.length && <div className="library-empty"><UiIcon name="notes" size={32} />
      <h2>A place for your next note</h2><p>Use New to write, generate notes, or add a document.</p></div>}
    {!!inbox?.files?.length && <section className="library-section" aria-label="Notes and files"><h2>Notes and files</h2>
      <div className="library-rows">{inbox.files.map(file => <article className="library-row" key={file.id}>
        <button className="library-open" onClick={() => openFile(file, inbox)}><span className="library-file-icon"><UiIcon name={file.type === "text/youtube" ? "link" : "notes"} /></span>
          <span><strong>{file.name}</strong><small>{file.type === "text/youtube" ? "Link" : file._isBlank ? "Note" : "Document"}{file.uploadedAt ? ` · ${file.uploadedAt}` : ""}</small></span></button>
        <ActionMenu label={`Actions for ${file.name}`} items={fileItems(file, inbox)} />
      </article>)}</div>
    </section>}
    {!!subjects.length && <section className="library-section" aria-label="Folders"><h2>Folders</h2><div className="library-rows">
      {subjects.map(folder => <article className="library-row" key={folder.id}>
        <button className="library-open" onClick={() => openFolder(folder)}><span className="library-file-icon"><UiIcon name="folder" /></span>
          <span><strong>{folder.name}</strong><small>{folder.files?.length || 0} files</small></span></button>
        <ActionMenu label={`Actions for ${folder.name}`} items={[
          { label: "Rename", run: () => rename({ ...folder, type: "folder" }) },
          { label: "Delete", danger: true, run: () => remove({ ...folder, type: "folder" }) },
        ]} />
      </article>)}
    </div></section>}
  </section>;
}
