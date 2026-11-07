import "./workspace.css";

export default function WorkspaceFrame() {
  return <div className="workspace-frame" aria-busy="true">
    <header><img src="/classio-mark.svg" width="28" height="28" alt="" /><span>Classio</span></header>
    <main><h1>Library</h1><div className="workspace-progress" role="progressbar" aria-label="Preparing workspace" /></main>
  </div>;
}
