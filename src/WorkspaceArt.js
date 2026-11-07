export default function WorkspaceArt({ kind = "library", className = "", ...props }) {
  return <svg viewBox="0 0 240 160" className={`workspace-art ${className}`} fill="none" aria-hidden="true" {...props}>
    <ellipse cx="122" cy="140" rx="85" ry="9" fill="currentColor" opacity=".07" />
    <path d="M39 47 104 30l18 89-65 17Z" fill="var(--art-paper,#fff)" stroke="currentColor" strokeWidth="2" />
    <path d="m56 66 37-10m-33 23 37-9m-33 23 27-7" stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity=".35" />
    <rect x="93" y="24" width="90" height="112" rx="12" fill="var(--app-accent)" stroke="currentColor" strokeWidth="2" />
    <path d="M105 24v112" stroke="currentColor" strokeWidth="2" opacity=".3" />
    {kind === "audio" ? <g stroke="currentColor" strokeWidth="4" strokeLinecap="round"><path d="M122 69v23m10-37v51m10-43v35m10-51v65m10-36v21" /></g>
      : <><path d="m122 62 13-7 12 7 14-7v35l-14 7-12-7-13 7Z" fill="var(--art-paper,#fff)" stroke="currentColor" strokeWidth="2" /><path d="M135 55v35m12-28v35" stroke="currentColor" strokeWidth="2" /></>}
    <circle cx="187" cy="44" r="17" fill="var(--art-paper,#fff)" stroke="currentColor" strokeWidth="2" />
    <path d="m180 44 5 5 10-11M38 26v10m-5-5h10m155 69v12m-6-6h12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
    <path d="m164 128 39-40 6 6-39 40-10 3Z" fill="#e7bb71" stroke="currentColor" strokeWidth="2" />
  </svg>;
}
