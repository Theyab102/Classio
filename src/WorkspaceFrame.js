import "./workspace.css";

export default function WorkspaceFrame() {
  return <div className="workspace-frame" aria-busy="true">
    <header><img src="/ClassioLogo-transparent.png" width="28" height="28" alt="" /><span>Classio</span></header>
    <div className="workspace-skeleton" role="status"><span className="sr-only">Preparing your study workspace…</span><aside aria-hidden="true"><i/><i/><i/><i/></aside><main aria-hidden="true"><div className="skeleton-title"/><div className="skeleton-line"/><div className="skeleton-actions">{[0,1,2,3].map(n=><i key={n}/>)}</div><div className="skeleton-line"/><div className="skeleton-materials">{[0,1,2].map(n=><i key={n}/>)}</div></main></div>
  </div>;
}
