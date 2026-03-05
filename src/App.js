import { useState, useRef, useEffect, useCallback } from "react";
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, onAuthStateChanged } from "firebase/auth";
import { getFirestore, doc, setDoc, getDoc } from "firebase/firestore";

// ─── FIREBASE ─────────────────────────────────────────────────────────────────
const firebaseConfig = {
  apiKey: "AIzaSyCDDXRAxLwNmzSYxs0HnnjV2TfhbqSdiag",
  authDomain: "classio-4378f.firebaseapp.com",
  projectId: "classio-4378f",
  storageBucket: "classio-4378f.firebasestorage.app",
  messagingSenderId: "595968221954",
  appId: "1:595968221954:web:cdefee80f05999f8bf181b",
};
const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const db = getFirestore(firebaseApp);
const googleProvider = new GoogleAuthProvider();

// ─── GROQ AI (FREE) ──────────────────────────────────────────────────────────
// Store your key in .env as REACT_APP_GROQ_KEY=gsk_...
// Get a free key at console.groq.com — no credit card needed!
const GROQ_KEY = process.env.REACT_APP_GROQ_KEY || "";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

async function callClaude(system, userMessage) {
  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${GROQ_KEY}` },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      messages: [{ role: "system", content: system }, { role: "user", content: userMessage }],
      max_tokens: 1200,
    }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.choices?.[0]?.message?.content || "";
}

async function callClaudeChat(system, messages) {
  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${GROQ_KEY}` },
    body: JSON.stringify({
      model: "llama-3.3-70b-versatile",
      messages: [{ role: "system", content: system }, ...messages],
      max_tokens: 1200,
    }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.choices?.[0]?.message?.content || "";
}

// Read a file as base64
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Read a text-based file as plain text
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

// Extract as much text as possible from any file type
async function extractFileText(fileObj) {
  if (!fileObj) return null;
  const name = fileObj.name.toLowerCase();
  const type = fileObj.type || "";

  // Plain text files
  if (type.startsWith("text/") || name.endsWith(".txt") || name.endsWith(".md") || name.endsWith(".csv")) {
    try { return await readFileAsText(fileObj); } catch { return null; }
  }

  // PDF — use PDF.js from CDN to extract text
  if (type === "application/pdf" || name.endsWith(".pdf")) {
    try {
      const base64 = await readFileAsBase64(fileObj);
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      // Dynamically load PDF.js
      if (!window.pdfjsLib) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
          "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      }

      const pdf = await window.pdfjsLib.getDocument({ data: bytes }).promise;
      let text = "";
      for (let i = 1; i <= Math.min(pdf.numPages, 20); i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map(item => item.str).join(" ") + "\n";
      }
      return text.trim() || null;
    } catch(e) { console.error("PDF read error", e); return null; }
  }

  // Images — return a note that it's an image
  if (type.startsWith("image/")) {
    return `[This is an image file: ${fileObj.name}. Describe its likely content based on the filename.]`;
  }

  // PowerPoint / Word / Excel — try to read as binary and extract any readable text
  if (name.endsWith(".pptx") || name.endsWith(".docx") || name.endsWith(".xlsx") ||
      name.endsWith(".ppt") || name.endsWith(".doc") || name.endsWith(".xls")) {
    try {
      // Load JSZip to unzip Office files
      if (!window.JSZip) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
      }
      const arrayBuffer = await fileObj.arrayBuffer();
      const zip = await window.JSZip.loadAsync(arrayBuffer);
      let text = "";

      // Extract text from all XML files inside the Office zip
      const xmlFiles = Object.keys(zip.files).filter(f =>
        f.endsWith(".xml") && (
          f.includes("slide") || f.includes("word/document") ||
          f.includes("sharedStrings") || f.includes("content")
        )
      );

      for (const xmlFile of xmlFiles.slice(0, 30)) {
        const xmlContent = await zip.files[xmlFile].async("string");
        // Strip XML tags and get just the text
        const stripped = xmlContent.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
        if (stripped.length > 20) text += stripped + "\n";
      }

      return text.slice(0, 8000) || null;
    } catch(e) { console.error("Office file read error", e); return null; }
  }

  return null;
}

// ─── COLORS ───────────────────────────────────────────────────────────────────
const C = {
  bg: "#F7F5F2", surface: "#FFFFFF", border: "#E8E4DF", text: "#1A1714",
  muted: "#9B9590", accent: "#3D5A80", accentL: "#E8EFF5", accentS: "#C5D5E8",
  warm: "#C17F5A", warmL: "#F5EDE5", green: "#4A7C59", greenL: "#E5F0E8",
  purple: "#6B4E8A", purpleL: "#EDE5F5", red: "#C45C5C", redL: "#F5E5E5",
};

const FILE_COLORS = [
  { bg: "#E8EFF5", accent: "#3D5A80" }, { bg: "#F5EDE5", accent: "#C17F5A" },
  { bg: "#E5F0E8", accent: "#4A7C59" }, { bg: "#EDE5F5", accent: "#6B4E8A" },
  { bg: "#F5E5E5", accent: "#C45C5C" }, { bg: "#EAEDF0", accent: "#4A5568" },
];
const FOLDER_COLORS = ["#3D5A80","#C17F5A","#4A7C59","#6B4E8A","#C45C5C","#4A5568","#8A7C4E","#4E7C8A"];

// ─── ICONS ────────────────────────────────────────────────────────────────────
const Icon = ({ d, size = 18, color = "currentColor", sw = 1.7 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
  </svg>
);
const I = {
  folder: "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
  file: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6"],
  plus: "M12 5v14M5 12h14",
  back: "M19 12H5M12 19l-7-7 7-7",
  ai: "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zM8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01",
  cards: "M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 1-4 4v14a3 3 0 0 0 3-3h7z",
  notes: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6","M16 13H8","M16 17H8","M10 9H8"],
  game: "M6 3v11.5A2.5 2.5 0 0 0 8.5 17h.5M18 3v11.5A2.5 2.5 0 0 1 15.5 17h-.5M8.5 17a2.5 2.5 0 0 0 0 5 2.5 2.5 0 0 0 0-5zM15.5 17a2.5 2.5 0 0 0 0 5 2.5 2.5 0 0 0 0-5z",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  send: "M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z",
  trash: "M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2",
  edit: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z",
  check: "M20 6L9 17l-5-5",
  sparkle: "M12 3L14.5 8.5H20L15.5 12L17 18L12 14.5L7 18L8.5 12L4 8.5H9.5L12 3Z",
  link: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",
  x: "M18 6L6 18M6 6l12 12",
  chevron: "M9 18l6-6-6-6",
  refresh: "M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15",
  paperclip: "M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48",
};

// ─── TEXT FORMATTER ───────────────────────────────────────────────────────────
const Fmt = ({ text }) => (
  <div>{(text||"").split('\n').map((line, i) => {
    const clean = line.replace(/^#+\s*/, "").replace(/\*\*/g, "");
    if (!line.trim()) return <br key={i} />;
    if (line.startsWith('# ') || line.startsWith('## ') || line.startsWith('### ')) return <p key={i} style={{ fontWeight:700, fontSize:15, marginBottom:6, marginTop:10, color:"#1a202c" }}>{clean}</p>;
    if (/^\*\*[^*]+\*\*$/.test(line.trim())) return <p key={i} style={{ fontWeight:700, fontSize:14, marginBottom:5, marginTop:8, color:"#2d3748" }}>{clean}</p>;
    if (line.startsWith('• ') || line.startsWith('- ') || line.startsWith('* ')) return <p key={i} style={{ paddingLeft:16, marginBottom:3, display:"flex", gap:6 }}><span style={{flexShrink:0}}>•</span><span>{line.slice(2).replace(/\*\*/g,"")}</span></p>;
    if (/^\d+\.\s/.test(line)) return <p key={i} style={{ paddingLeft:16, marginBottom:3 }}>{clean}</p>;
    const parts = line.split(/(\*\*[^*]+\*\*)/g);
    return <p key={i} style={{ marginBottom:3, lineHeight:1.6 }}>{parts.map((p,j) => p.startsWith('**') ? <strong key={j}>{p.slice(2,-2)}</strong> : p)}</p>;
  })}</div>
);

// ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
const GS = `@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Fraunces:wght@400;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0} input,textarea,button{font-family:inherit}
::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-thumb{background:#D8D4CF;border-radius:3px}
.hov:hover{opacity:0.82} .card-hov:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.10)!important}
.card-hov{transition:all .2s} .tab:hover{background:#F0EDE9!important} .row:hover{background:#F7F5F2!important} .row{transition:background .15s}
@media(max-width:768px){
  .desktop-only{display:none!important}
  .mobile-stack{flex-direction:column!important}
  .mobile-full{width:100%!important;max-width:100%!important}
  .mobile-pad{padding:12px 14px!important}
  .mobile-small-text{font-size:12px!important}
  .tab-label{display:none}
  .ad-bar{height:44px!important}
  .ad-bar ins{height:36px!important}
  .toolbar-wrap{flex-wrap:wrap;gap:6px!important}
}
@keyframes bounce{0%,80%,100%{transform:scale(.8);opacity:.5}40%{transform:scale(1.1);opacity:1}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
`; 

// Global file object store — survives navigation within the session
const FILE_STORE = new Map();

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(undefined);
  const [isGuest, setIsGuest] = useState(false);
  const [guestName, setGuestName] = useState("");
  const [folders, setFolders] = useState([]);
  const [screen, setScreen] = useState("home");
  const [activeFolder, setActiveFolder] = useState(null);
  const [activeFile, setActiveFile] = useState(null);
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [newName, setNewName] = useState("");
  const [newColor, setNewColor] = useState(FOLDER_COLORS[0]);
  const [saving, setSaving] = useState(false);
  const saveTimer = useRef(null);

  useEffect(() => {
    return onAuthStateChanged(auth, async (u) => {
      setUser(u);
      if (u) {
        setIsGuest(false);
        const snap = await getDoc(doc(db, "users", u.uid)).catch(() => null);
        if (snap?.exists()) setFolders(snap.data().folders || []);
        else setFolders([]);
      }
    });
  }, []);

  const save = useCallback((flds, uid) => {
    if (!uid) return;
    clearTimeout(saveTimer.current);
    setSaving(true);
    saveTimer.current = setTimeout(async () => {
      const clean = flds.map(f => ({ ...f, files: f.files.map(({ _fileObj, linkedFiles, ...rest }) => ({
        ...rest,
        linkedFiles: (linkedFiles || []).map(({ _fileObj: _lfo, ...lr }) => lr),
      }))}));
      await setDoc(doc(db, "users", uid), { folders: clean }, { merge: true }).catch(console.error);
      setSaving(false);
    }, 800);
  }, []);

  const setFoldersSave = (flds) => {
    setFolders(flds);
    if (user && !isGuest) save(flds, user.uid);
  };

  const updateFolder = (updated) => {
    const next = folders.map(f => f.id === updated.id ? updated : f);
    setFoldersSave(next);
    if (activeFolder?.id === updated.id) setActiveFolder(updated);
  };

  const updateFile = (folderId, updated) => {
    // Always carry _fileObj forward so viewer never loses it
    const withObj = { ...updated, _fileObj: updated._fileObj || FILE_STORE.get(updated.id) || null };
    const next = folders.map(f => f.id === folderId
      ? { ...f, files: f.files.map(fi => fi.id === withObj.id ? withObj : fi) }
      : f);
    setFoldersSave(next);
    setActiveFile(withObj);
    setActiveFolder(prev => prev ? { ...prev, files: prev.files.map(fi => fi.id === withObj.id ? withObj : fi) } : prev);
  };

  const handleGuest = (name) => { setGuestName(name); setIsGuest(true); setFolders([]); };

  const handleGuestSignOut = () => {
    setIsGuest(false); setGuestName(""); setFolders([]);
    setScreen("home"); setActiveFolder(null); setActiveFile(null);
  };

  if (user === undefined && !isGuest) return <Splash />;
  if (!user && !isGuest) return <SignIn
    onSignIn={() => signInWithPopup(auth, googleProvider).catch(console.error)}
    onGuest={handleGuest} />;

  if (screen === "file" && activeFile && activeFolder) {
    return <FileView file={activeFile} folder={activeFolder} allFiles={activeFolder.files}
      user={user} isGuest={isGuest}
      onBack={() => { setScreen("folder"); setActiveFile(null); }}
      onUpdate={(u) => updateFile(activeFolder.id, u)} />;
  }

  if (screen === "folder" && activeFolder) {
    const folder = folders.find(f => f.id === activeFolder.id) || activeFolder;
    return <FolderView folder={folder} onBack={() => { setScreen("home"); setActiveFolder(null); }}
      onOpenFile={(f) => { const restored = {...f, _fileObj: f._fileObj || FILE_STORE.get(f.id) || null}; setActiveFile(restored); setScreen("file"); }}
      onUpdate={updateFolder} />;
  }

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "'DM Sans', sans-serif", paddingBottom: 52 }}>
      <style>{GS}</style>
      <Header user={isGuest ? { displayName: guestName, photoURL: null } : user} saving={saving} isGuest={isGuest} onSignOut={isGuest ? handleGuestSignOut : () => signOut(auth)} />
      <AdBanner />
      <div style={{ maxWidth: 900, margin: "0 auto", padding: "24px 14px" }}>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom: 32 }}>
          <div>
            <h1 style={{ fontFamily:"'Fraunces',serif", fontSize: 32, fontWeight: 700, color: C.text, letterSpacing: -1 }}>My Folders</h1>
            <p style={{ fontSize: 14, color: C.muted, marginTop: 4 }}>{folders.length === 0 ? "Create your first folder to get started" : `${folders.length} folder${folders.length !== 1 ? "s" : ""}`}</p>
          </div>
          <button onClick={() => setShowNewFolder(true)} className="hov"
            style={{ display:"flex", alignItems:"center", gap: 8, background: C.accent, color:"#fff", border:"none", borderRadius: 12, padding:"10px 20px", fontSize: 14, fontWeight: 600, cursor:"pointer" }}>
            <Icon d={I.plus} size={16} color="#fff" sw={2.5} /> New Folder
          </button>
        </div>

        {folders.length === 0 && (
          <div style={{ textAlign:"center", padding:"80px 0" }}>
            <div style={{ width:80, height:80, background: C.accentL, borderRadius:24, display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 20px" }}>
              <Icon d={I.folder} size={36} color={C.accent} />
            </div>
            <p style={{ fontSize:18, fontWeight:600, color:C.text, marginBottom:8 }}>No folders yet</p>
            <p style={{ fontSize:14, color:C.muted, maxWidth:280, margin:"0 auto 24px" }}>Create a folder for each subject to organise your files</p>
            <button onClick={() => setShowNewFolder(true)} className="hov"
              style={{ background:C.accent, color:"#fff", border:"none", borderRadius:10, padding:"10px 24px", fontSize:14, fontWeight:600, cursor:"pointer" }}>
              Create First Folder
            </button>
          </div>
        )}

        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(220px,1fr))", gap:16 }}>
          {folders.map(folder => (
            <div key={folder.id} className="card-hov"
              onClick={() => { setActiveFolder(folder); setScreen("folder"); }}
              style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:16, padding:20, cursor:"pointer", boxShadow:"0 2px 8px rgba(0,0,0,.05)" }}>
              <div style={{ width:44, height:44, background:folder.color+"22", borderRadius:12, display:"flex", alignItems:"center", justifyContent:"center", marginBottom:14 }}>
                <Icon d={I.folder} size={22} color={folder.color} />
              </div>
              <p style={{ fontSize:15, fontWeight:600, color:C.text, marginBottom:4, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{folder.name}</p>
              <p style={{ fontSize:13, color:C.muted }}>{folder.files.length} file{folder.files.length !== 1 ? "s" : ""}</p>
            </div>
          ))}
        </div>
      </div>

      {showNewFolder && (
        <Modal onClose={() => { setShowNewFolder(false); setNewName(""); }}>
          <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:22, fontWeight:700, color:C.text, marginBottom:20 }}>New Folder</h2>
          <label style={{ fontSize:13, fontWeight:600, color:C.muted, display:"block", marginBottom:6 }}>NAME</label>
          <input autoFocus value={newName} onChange={e => setNewName(e.target.value)}
            onKeyDown={e => { if (e.key==="Enter" && newName.trim()) { setFoldersSave([...folders,{id:`f${Date.now()}`,name:newName.trim(),color:newColor,files:[]}]); setShowNewFolder(false); setNewName(""); }}}
            placeholder="e.g. Biology, Maths…"
            style={{ width:"100%", border:`1.5px solid ${C.border}`, borderRadius:10, padding:"10px 14px", fontSize:15, outline:"none", marginBottom:16, color:C.text, background:C.bg }} />
          <label style={{ fontSize:13, fontWeight:600, color:C.muted, display:"block", marginBottom:10 }}>COLOUR</label>
          <div style={{ display:"flex", gap:8, marginBottom:24 }}>
            {FOLDER_COLORS.map(col => <button key={col} onClick={() => setNewColor(col)} style={{ width:28, height:28, borderRadius:"50%", background:col, border:newColor===col?`3px solid ${C.text}`:"3px solid transparent", cursor:"pointer" }} />)}
          </div>
          <div style={{ display:"flex", gap:10 }}>
            <button onClick={() => { setShowNewFolder(false); setNewName(""); }}
              style={{ flex:1, padding:"10px", border:`1.5px solid ${C.border}`, borderRadius:10, background:"transparent", fontSize:14, fontWeight:600, cursor:"pointer", color:C.text }}>Cancel</button>
            <button disabled={!newName.trim()}
              onClick={() => { setFoldersSave([...folders,{id:`f${Date.now()}`,name:newName.trim(),color:newColor,files:[]}]); setShowNewFolder(false); setNewName(""); }}
              style={{ flex:2, padding:"10px", background:newName.trim()?C.accent:C.border, color:newName.trim()?"#fff":C.muted, border:"none", borderRadius:10, fontSize:14, fontWeight:600, cursor:newName.trim()?"pointer":"not-allowed" }}>
              Create Folder
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

// ─── AD BANNER ───────────────────────────────────────────────────────────────
function AdBanner() {
  useEffect(() => {
    if (!document.querySelector('script[src*="adsbygoogle"]')) {
      const script = document.createElement("script");
      script.src = "https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5802600279565250";
      script.async = true;
      script.crossOrigin = "anonymous";
      document.head.appendChild(script);
    }
    try { (window.adsbygoogle = window.adsbygoogle || []).push({}); } catch(e) {}
  }, []);

  return (
    <div className="ad-bar" style={{ position:"fixed", bottom:0, left:0, right:0, zIndex:999, background:C.surface, borderTop:`1px solid ${C.border}`, display:"flex", alignItems:"center", justifyContent:"center", padding:"3px 8px", height:50 }}>
      <div style={{ position:"relative", width:"100%", maxWidth:728, height:42 }}>
        <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center", justifyContent:"center", background:C.bg, border:`1px dashed ${C.border}`, borderRadius:6, pointerEvents:"none" }}>
          <span style={{ fontSize:10, color:C.muted, letterSpacing:1 }}>ADVERTISEMENT</span>
        </div>
        <ins className="adsbygoogle"
          style={{ display:"block", width:"100%", height:42, position:"relative", zIndex:1 }}
          data-ad-client="ca-pub-5802600279565250"
          data-ad-slot="7527000448"
          data-ad-format="auto"
          data-full-width-responsive="true" />
      </div>
    </div>
  );
}

// ─── HEADER ───────────────────────────────────────────────────────────────────
function Header({ user, saving, isGuest, onSignOut }) {
  return (
    <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 16px", height:56, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <div style={{ width:30, height:30, background:C.accent, borderRadius:9, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <Icon d={I.sparkle} size={15} color="#fff" sw={2} />
        </div>
        <span style={{ fontFamily:"'Fraunces',serif", fontSize:20, fontWeight:700, color:C.text, letterSpacing:-0.5 }}>Classio</span>
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        {!isGuest && <span className="desktop-only" style={{ fontSize:12, color: saving ? C.muted : C.green }}>{saving ? "Saving…" : "✓ Saved"}</span>}
        {isGuest && <span className="desktop-only" style={{ fontSize:11, background:C.warmL, color:C.warm, border:`1px solid ${C.warm}44`, borderRadius:20, padding:"3px 8px", fontWeight:600 }}>Guest</span>}
        {user?.photoURL
          ? <img src={user.photoURL} alt="" style={{ width:30, height:30, borderRadius:"50%", border:`2px solid ${C.border}`, flexShrink:0 }} />
          : <div style={{ width:30, height:30, background:isGuest?C.warmL:C.accentL, borderRadius:"50%", display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, fontWeight:700, color:isGuest?C.warm:C.accent, flexShrink:0 }}>{user?.displayName?.[0]?.toUpperCase() || "G"}</div>
        }
        <span className="desktop-only" style={{ fontSize:13, fontWeight:600, color:C.text }}>{isGuest ? user?.displayName : user?.displayName?.split(" ")[0]}</span>
        <button onClick={onSignOut} className="hov"
          style={{ fontSize:12, color:C.muted, background:"none", border:`1px solid ${C.border}`, borderRadius:7, padding:"4px 9px", cursor:"pointer", whiteSpace:"nowrap" }}>{isGuest ? "Exit" : "Sign out"}</button>
      </div>
    </div>
  );
}

// ─── SPLASH / SIGN IN ─────────────────────────────────────────────────────────
function Splash() {
  return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", alignItems:"center", justifyContent:"center" }}>
      <style>{GS}</style>
      <div style={{ textAlign:"center" }}>
        <div style={{ width:56, height:56, background:C.accent, borderRadius:18, display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 16px" }}>
          <Icon d={I.sparkle} size={24} color="#fff" sw={2} />
        </div>
        <p style={{ fontSize:15, color:C.muted, fontFamily:"'DM Sans',sans-serif" }}>Loading…</p>
      </div>
    </div>
  );
}

function SignIn({ onSignIn, onGuest }) {
  const [showGuest, setShowGuest] = useState(false);
  const [name, setName] = useState("");

  if (showGuest) return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", alignItems:"center", justifyContent:"center", fontFamily:"'DM Sans',sans-serif" }}>
      <style>{GS}</style>
      <div style={{ background:C.surface, borderRadius:28, padding:"56px 48px", width:"100%", maxWidth:440, boxShadow:"0 8px 40px rgba(0,0,0,.08)", textAlign:"center" }}>
        <div style={{ width:68, height:68, background:C.warmL, borderRadius:22, display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 24px" }}>
          <span style={{ fontSize:32 }}>👤</span>
        </div>
        <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:28, fontWeight:700, color:C.text, marginBottom:8 }}>Continue as Guest</h2>
        <p style={{ fontSize:14, color:C.muted, marginBottom:28, lineHeight:1.6 }}>Your folders will not be saved when you leave.<br/>Sign in with Google to keep your data.</p>
        <input autoFocus value={name} onChange={e => setName(e.target.value)}
          onKeyDown={e => { if(e.key==="Enter" && name.trim()) onGuest(name.trim()); }}
          placeholder="Enter your name…"
          style={{ width:"100%", border:`1.5px solid ${C.border}`, borderRadius:12, padding:"12px 16px", fontSize:15, outline:"none", marginBottom:14, color:C.text, background:C.bg, textAlign:"center" }} />
        <button disabled={!name.trim()} onClick={() => name.trim() && onGuest(name.trim())}
          style={{ width:"100%", background:name.trim()?C.warm:"#ccc", color:"#fff", border:"none", borderRadius:12, padding:"13px", fontSize:15, fontWeight:700, cursor:name.trim()?"pointer":"not-allowed", marginBottom:12 }}>
          Enter as Guest
        </button>
        <button onClick={() => setShowGuest(false)}
          style={{ width:"100%", background:"none", border:"none", color:C.muted, fontSize:14, cursor:"pointer" }}>
          ← Back
        </button>
      </div>
    </div>
  );

  return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", alignItems:"center", justifyContent:"center", fontFamily:"'DM Sans',sans-serif" }}>
      <style>{GS}</style>
      <div style={{ background:C.surface, borderRadius:28, padding:"56px 48px", width:"100%", maxWidth:440, boxShadow:"0 8px 40px rgba(0,0,0,.08)", textAlign:"center" }}>
        <div style={{ width:68, height:68, background:C.accentL, borderRadius:22, display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 24px" }}>
          <Icon d={I.sparkle} size={30} color={C.accent} sw={2} />
        </div>
        <h1 style={{ fontFamily:"'Fraunces',serif", fontSize:34, fontWeight:700, color:C.text, letterSpacing:-0.5, marginBottom:10 }}>Classio</h1>
        <p style={{ fontSize:15, color:C.muted, marginBottom:40, lineHeight:1.6 }}>Your AI-powered study space.<br/>Sign in to save across devices.</p>
        <button onClick={onSignIn} className="hov"
          style={{ width:"100%", display:"flex", alignItems:"center", justifyContent:"center", gap:12, background:C.surface, border:`1.5px solid ${C.border}`, borderRadius:14, padding:"14px 20px", fontSize:15, fontWeight:600, cursor:"pointer", color:C.text, boxShadow:"0 2px 8px rgba(0,0,0,.06)", marginBottom:14 }}>
          <svg width="20" height="20" viewBox="0 0 48 48">
            <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
            <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
            <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
            <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
          </svg>
          Continue with Google
        </button>
        <div style={{ display:"flex", alignItems:"center", gap:12, margin:"4px 0 14px" }}>
          <div style={{ flex:1, height:1, background:C.border }} />
          <span style={{ fontSize:13, color:C.muted }}>or</span>
          <div style={{ flex:1, height:1, background:C.border }} />
        </div>
        <button onClick={() => setShowGuest(true)} className="hov"
          style={{ width:"100%", background:"transparent", border:`1.5px solid ${C.border}`, borderRadius:14, padding:"13px 20px", fontSize:15, fontWeight:600, cursor:"pointer", color:C.muted }}>
          Continue as Guest
        </button>
        <p style={{ fontSize:12, color:C.muted, marginTop:16, lineHeight:1.5 }}>Guest mode does not save your data between sessions.</p>
      </div>
    </div>
  );
}

// ─── FOLDER VIEW ──────────────────────────────────────────────────────────────
function FolderView({ folder, onBack, onOpenFile, onUpdate }) {
  const [dragging, setDragging] = useState(false);
  const [tab, setTab] = useState("files");
  const [editingName, setEditingName] = useState(false);
  const [folderName, setFolderName] = useState(folder.name);
  const fileInput = useRef();

  const addFiles = (list) => {
    const added = Array.from(list).map(f => {
      const id = `fi${Date.now()}-${Math.random()}`;
      FILE_STORE.set(id, f); // persist file object globally
      return {
        id, name: f.name, type: f.type, size: f.size,
        colorIndex: 0, notes: "", studyCards: [], uploadedAt: new Date().toLocaleDateString(),
        linkedFiles: [], _fileObj: f,
      };
    });
    onUpdate({ ...folder, files: [...folder.files, ...added] });
  };

  const TABS = [{ id:"files", label:"Files", icon:I.file },{ id:"ai", label:"AI Assistant", icon:I.ai }];

  return (
    <div style={{ minHeight:"100vh", background:C.bg, fontFamily:"'DM Sans',sans-serif" }}>
      <style>{GS}</style>
      {/* Top bar */}
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 24px", height:64, display:"flex", alignItems:"center", gap:16 }}>
        <button onClick={onBack} className="hov" style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14 }}>
          <Icon d={I.back} size={18} color={C.muted} /> Back
        </button>
        <div style={{ width:1, height:20, background:C.border }} />
        <div style={{ display:"flex", alignItems:"center", gap:10, flex:1 }}>
          <div style={{ width:34, height:34, background:folder.color+"22", borderRadius:10, display:"flex", alignItems:"center", justifyContent:"center" }}>
            <Icon d={I.folder} size={18} color={folder.color} />
          </div>
          {editingName
            ? <input autoFocus value={folderName} onChange={e => setFolderName(e.target.value)}
                onBlur={() => { setEditingName(false); onUpdate({...folder,name:folderName||folder.name}); }}
                onKeyDown={e => { if(e.key==="Enter"){setEditingName(false);onUpdate({...folder,name:folderName||folder.name});} }}
                style={{ fontSize:18, fontWeight:700, fontFamily:"'Fraunces',serif", border:"none", borderBottom:`2px solid ${C.accent}`, outline:"none", background:"transparent", color:C.text, width:240 }} />
            : <h1 onClick={() => setEditingName(true)} style={{ fontFamily:"'Fraunces',serif", fontSize:20, fontWeight:700, color:C.text, cursor:"text" }} title="Click to rename">{folder.name}</h1>
          }
        </div>
        <div style={{ display:"flex", gap:6 }}>
          {FOLDER_COLORS.map(col => <button key={col} onClick={() => onUpdate({...folder,color:col})} style={{ width:20, height:20, borderRadius:"50%", background:col, border:folder.color===col?`3px solid ${C.text}`:"2px solid transparent", cursor:"pointer" }} />)}
        </div>
      </div>
      {/* Tabs */}
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 12px", display:"flex", gap:2, overflowX:"auto" }}>
        {TABS.map(t => (
          <button key={t.id} className="tab" onClick={() => setTab(t.id)}
            style={{ display:"flex", alignItems:"center", gap:6, padding:"12px 14px", border:"none", borderBottom:tab===t.id?`2px solid ${C.accent}`:"2px solid transparent", background:"none", cursor:"pointer", fontSize:13, fontWeight:tab===t.id?700:500, color:tab===t.id?C.accent:C.muted, marginBottom:-1, whiteSpace:"nowrap", flexShrink:0 }}>
            <Icon d={t.icon} size={14} color={tab===t.id?C.accent:C.muted} />
            <span className="tab-label">{t.label}</span>
          </button>
        ))}
      </div>

      <div style={{ maxWidth:860, margin:"0 auto", padding:"32px 24px" }}>
        {tab === "files" && (
          <>
            <div onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)}
              onDrop={e=>{e.preventDefault();setDragging(false);addFiles(e.dataTransfer.files);}}
              onClick={()=>fileInput.current.click()}
              style={{ border:`2px dashed ${dragging?C.accent:C.border}`, borderRadius:16, padding:"28px", textAlign:"center", cursor:"pointer", background:dragging?C.accentL:"transparent", marginBottom:24, transition:"all .2s" }}>
              <Icon d={I.upload} size={28} color={dragging?C.accent:C.muted} />
              <p style={{ fontSize:15, fontWeight:600, color:dragging?C.accent:C.text, marginTop:10, marginBottom:4 }}>Drop files here or click to upload</p>
              <p style={{ fontSize:13, color:C.muted }}>PDF, Word, PowerPoint, images, and more</p>
              <input ref={fileInput} type="file" multiple style={{ display:"none" }} onChange={e=>addFiles(e.target.files)} />
            </div>

            {folder.files.length === 0
              ? <div style={{ textAlign:"center", padding:"40px 0", color:C.muted }}><Icon d={I.file} size={40} color={C.border} /><p style={{ marginTop:12, fontSize:15 }}>No files yet</p></div>
              : <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
                  {folder.files.map(file => {
                    const fc = FILE_COLORS[file.colorIndex||0];
                    const linked = file.linkedFiles || [];
                    return (
                      <div key={file.id} className="row"
                        style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:12, padding:"12px 16px" }}>
                        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
                          <div style={{ width:38, height:38, background:fc.bg, borderRadius:10, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
                            <Icon d={I.file} size={18} color={fc.accent} />
                          </div>
                          <div style={{ flex:1, minWidth:0 }}>
                            <p style={{ fontSize:14, fontWeight:600, color:C.text, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{file.name}</p>
                            <p style={{ fontSize:12, color:C.muted }}>
                              {(file.size/1024).toFixed(1)} KB · {file.uploadedAt}
                              {linked.length > 0 && <span style={{ marginLeft:8, color:C.accent }}>🔗 {linked.length} linked</span>}
                            </p>
                          </div>
                          <div style={{ display:"flex", gap:4 }}>
                            {FILE_COLORS.map((col,ci) => (
                              <button key={ci} onClick={() => onUpdate({...folder,files:folder.files.map(f=>f.id===file.id?{...f,colorIndex:ci}:f)})}
                                style={{ width:14, height:14, borderRadius:"50%", background:col.accent, border:file.colorIndex===ci?`2px solid ${C.text}`:"2px solid transparent", cursor:"pointer" }} />
                            ))}
                          </div>
                          <button onClick={() => onOpenFile(file)} className="hov"
                            style={{ display:"flex", alignItems:"center", gap:6, background:C.accentL, color:C.accent, border:"none", borderRadius:8, padding:"7px 14px", fontSize:13, fontWeight:600, cursor:"pointer" }}>
                            <Icon d={I.edit} size={13} color={C.accent} /> Open
                          </button>
                          <button onClick={() => onUpdate({...folder,files:folder.files.filter(f=>f.id!==file.id)})} className="hov"
                            style={{ background:"none", border:"none", cursor:"pointer", padding:4 }}>
                            <Icon d={I.trash} size={16} color={C.muted} />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
            }
          </>
        )}
        {tab === "ai" && <AITab file={null} allFiles={folder.files} folder={folder} onUpdate={()=>{}} />}
      </div>
    </div>
  );
}

// ─── FILE VIEW ────────────────────────────────────────────────────────────────
function FileView({ file, folder, allFiles, user, isGuest, onBack, onUpdate }) {
  const [tab, setTab] = useState("view");
  const TABS = [
    {id:"view",label:"View & Annotate",icon:I.file},
    {id:"notes",label:"Notes",icon:I.notes},
    {id:"cards",label:"Study Cards",icon:I.cards},
    {id:"ai",label:"AI Assistant",icon:I.ai},
    {id:"game",label:"Game Mode",icon:I.game},
  ];
  const fc = FILE_COLORS[file.colorIndex||0];

  return (
    <div style={{ minHeight:"100vh", background:C.bg, fontFamily:"'DM Sans',sans-serif" }}>
      <style>{GS}</style>
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 24px", height:64, display:"flex", alignItems:"center", gap:14 }}>
        <button onClick={onBack} className="hov" style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14 }}>
          <Icon d={I.back} size={18} color={C.muted} /> {folder.name}
        </button>
        <Icon d={I.chevron} size={14} color={C.border} />
        <div style={{ width:28, height:28, background:fc.bg, borderRadius:8, display:"flex", alignItems:"center", justifyContent:"center" }}>
          <Icon d={I.file} size={14} color={fc.accent} />
        </div>
        <span style={{ fontSize:15, fontWeight:600, color:C.text, flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{file.name}</span>
      </div>
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 24px", display:"flex", gap:4 }}>
        {TABS.map(t => (
          <button key={t.id} className="tab" onClick={() => setTab(t.id)}
            style={{ display:"flex", alignItems:"center", gap:7, padding:"14px 18px", border:"none", borderBottom:tab===t.id?`2px solid ${C.accent}`:"2px solid transparent", background:"none", cursor:"pointer", fontSize:14, fontWeight:tab===t.id?700:500, color:tab===t.id?C.accent:C.muted, marginBottom:-1 }}>
            <Icon d={t.icon} size={15} color={tab===t.id?C.accent:C.muted} />{t.label}
          </button>
        ))}
      </div>
      {tab==="view"
        ? <ViewTab file={file} onUpdate={onUpdate} />
        : <div style={{ maxWidth:900, margin:"0 auto", padding:"32px 24px" }}>
            {tab==="notes" && <NotesTab file={file} onUpdate={onUpdate} user={user} isGuest={isGuest} />}
            {tab==="cards" && <CardsTab file={file} onUpdate={onUpdate} />}
            {tab==="ai" && <AITab file={file} allFiles={allFiles} folder={folder} onUpdate={onUpdate} />}
            {tab==="game" && <GameTab file={file} />}
          </div>
      }
    </div>
  );
}

function ViewTab({ file, onUpdate }) {
  const canvasRef = useRef(null);
  const drawCanvasRef = useRef(null);
  const [pdfDoc, setPdfDoc] = useState(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [loading, setLoading] = useState(false);
  const [tool, setTool] = useState("pen");
  const [color, setColor] = useState("#E53E3E");
  const [brushSize, setBrushSize] = useState(3);
  const [drawing, setDrawing] = useState(false);
  const [lastPos, setLastPos] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const [explanation, setExplanation] = useState("");
  const [showExplain, setShowExplain] = useState(false);
  const [explainPageNum, setExplainPageNum] = useState(1);
  const [imgSrc, setImgSrc] = useState(null);
  const [annotations, setAnnotations] = useState({});
  // Resolve file object — check live reference first, then fall back to FILE_STORE
  const fileObj = file._fileObj || FILE_STORE.get(file.id) || null;

  const isPDF = fileObj && (fileObj.type === "application/pdf" || fileObj.name.toLowerCase().endsWith(".pdf"));
  const isImage = fileObj && fileObj.type.startsWith("image/");
  const isText = fileObj && (fileObj.type.startsWith("text/") || fileObj.name.endsWith(".txt") || fileObj.name.endsWith(".md") || fileObj.name.endsWith(".csv"));
  const isWord = fileObj && (fileObj.name.toLowerCase().endsWith(".docx") || fileObj.name.toLowerCase().endsWith(".doc"));
  const isPPT = fileObj && (fileObj.name.toLowerCase().endsWith(".pptx") || fileObj.name.toLowerCase().endsWith(".ppt"));
  const isExcel = fileObj && (fileObj.name.toLowerCase().endsWith(".xlsx") || fileObj.name.toLowerCase().endsWith(".xls"));
  const isOffice = isWord || isPPT || isExcel;

  // Load PDF
  useEffect(() => {
    if (!isPDF || !fileObj) return;
    setLoading(true);
    (async () => {
      if (!window.pdfjsLib) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
          "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      }
      const ab = await fileObj.arrayBuffer();
      const doc = await window.pdfjsLib.getDocument({ data: ab }).promise;
      setPdfDoc(doc);
      setTotalPages(doc.numPages);
      setLoading(false);
    })();
  }, [fileObj]);

  // Load image URL
  useEffect(() => {
    if (!isImage || !fileObj) return;
    const url = URL.createObjectURL(fileObj);
    setImgSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [fileObj]);

  // Save annotation for current page before switching
  const saveAnnotation = () => {
    if (drawCanvasRef.current) {
      setAnnotations(prev => ({ ...prev, [page]: drawCanvasRef.current.toDataURL() }));
    }
  };

  // Render PDF page onto canvas - with render task cancellation to prevent white/colored pages
  const renderTaskRef = useRef(null);
  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;
    // Cancel any in-progress render
    if (renderTaskRef.current) {
      renderTaskRef.current.cancel();
      renderTaskRef.current = null;
    }
    (async () => {
      try {
        const pg = await pdfDoc.getPage(page);
        const vp = pg.getViewport({ scale: 1.5 });
        const c = canvasRef.current;
        if (!c) return;
        c.width = vp.width;
        c.height = vp.height;
        const ctx = c.getContext("2d");
        // Fill white background first so page always looks correct
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, c.width, c.height);
        const task = pg.render({ canvasContext: ctx, viewport: vp });
        renderTaskRef.current = task;
        await task.promise;
        renderTaskRef.current = null;
        // Sync draw canvas
        if (drawCanvasRef.current) {
          const dc = drawCanvasRef.current;
          dc.width = vp.width;
          dc.height = vp.height;
          const dctx = dc.getContext("2d");
          dctx.clearRect(0, 0, dc.width, dc.height);
          if (annotations[page]) {
            const img = new Image();
            img.onload = () => dctx.drawImage(img, 0, 0);
            img.src = annotations[page];
          }
        }
      } catch(e) {
        if (e.name !== "RenderingCancelledException") console.error("PDF render error", e);
      }
    })();
  }, [pdfDoc, page]);

  const changePage = (n) => { saveAnnotation(); setPage(n); };

  // Drawing helpers
  const getPos = (e, canvas) => {
    const r = canvas.getBoundingClientRect();
    const sx = canvas.width / r.width, sy = canvas.height / r.height;
    const cx = e.touches ? e.touches[0].clientX : e.clientX;
    const cy = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (cx - r.left) * sx, y: (cy - r.top) * sy };
  };

  const startDraw = (e) => {
    e.preventDefault();
    if (!drawCanvasRef.current) return;
    setDrawing(true);
    setLastPos(getPos(e, drawCanvasRef.current));
  };

  const doDraw = (e) => {
    e.preventDefault();
    if (!drawing || !drawCanvasRef.current || !lastPos) return;
    const canvas = drawCanvasRef.current;
    const ctx = canvas.getContext("2d");
    const pos = getPos(e, canvas);
    ctx.beginPath();
    ctx.moveTo(lastPos.x, lastPos.y);
    ctx.lineTo(pos.x, pos.y);
    if (tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.lineWidth = brushSize * 6;
    } else if (tool === "highlight") {
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = brushSize * 8;
      ctx.strokeStyle = color;
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
      ctx.lineWidth = brushSize;
      ctx.strokeStyle = color;
    }
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
    setLastPos(pos);
  };

  const stopDraw = () => { if (drawing) { setDrawing(false); saveAnnotation(); } };

  const clearDrawing = () => {
    if (drawCanvasRef.current) {
      drawCanvasRef.current.getContext("2d").clearRect(0, 0, drawCanvasRef.current.width, drawCanvasRef.current.height);
      setAnnotations(prev => { const n = {...prev}; delete n[page]; return n; });
    }
  };

  const doExplain = async (pg) => {
    setExplaining(true); setShowExplain(true); setExplanation("");
    try {
      let pageText = "";
      if (pdfDoc) {
        const p = await pdfDoc.getPage(pg);
        const tc = await p.getTextContent();
        pageText = tc.items.map(i => i.str).join(" ");
      } else {
        const t = await extractFileText(fileObj).catch(() => "");
        pageText = (t || "").slice(0, 4000);
      }
      const txt = await callClaude(
        "You are a helpful study tutor. Explain content clearly for a student. Use **bold** for key terms and • for bullet points.",
        pageText
          ? `Explain this content from ${isPDF ? `page ${pg} of` : ""} "${file.name}" clearly:

${pageText.slice(0, 4000)}`
          : `Briefly explain what a student should know about this topic from "${file.name}".`
      );
      setExplanation(txt);
    } catch(e) { setExplanation(`Error: ${e.message}`); }
    setExplaining(false);
  };

  const COLORS = ["#E53E3E","#FF8C00","#ECC94B","#38A169","#3182CE","#805AD5","#000000","#FFFFFF"];
  const TOOLS = [{ id:"pen", label:"✏️" }, { id:"highlight", label:"🖊️" }, { id:"eraser", label:"🧹" }];

  if (!fileObj) return (
    <div style={{ textAlign:"center", padding:"60px 24px" }}>
      <div style={{ fontSize:48, marginBottom:12 }}>📂</div>
      <p style={{ fontSize:16, fontWeight:600, color:C.text, marginBottom:6 }}>Open file to view it</p>
      <p style={{ fontSize:13, color:C.muted, marginBottom:16 }}>This file needs to be opened from your device to display here.</p>
      <label style={{ display:"inline-flex", alignItems:"center", gap:8, background:C.accent, color:"#fff", borderRadius:10, padding:"10px 20px", cursor:"pointer", fontSize:14, fontWeight:600 }}>
        📁 Open File
        <input type="file" style={{ display:"none" }} onChange={e => {
          const f = e.target.files?.[0];
          if (f) { FILE_STORE.set(file.id, f); onUpdate({...file, _fileObj: f}); }
        }} />
      </label>
    </div>
  );

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"calc(100vh - 108px)" }}>
      {/* Toolbar */}
      <div className="toolbar-wrap" style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"7px 12px", display:"flex", alignItems:"center", gap:8, flexWrap:"wrap" }}>
        {/* Draw tools */}
        <div style={{ display:"flex", gap:4 }}>
          {TOOLS.map(t => (
            <button key={t.id} onClick={() => setTool(t.id)}
              style={{ width:32, height:32, borderRadius:7, border:`1.5px solid ${tool===t.id?C.accent:C.border}`, background:tool===t.id?C.accentL:"#fff", cursor:"pointer", fontSize:14 }}>
              {t.label}
            </button>
          ))}
        </div>
        <div style={{ width:1, height:20, background:C.border }} />
        {/* Colors */}
        <div style={{ display:"flex", gap:3 }}>
          {COLORS.map(col => (
            <button key={col} onClick={() => setColor(col)}
              style={{ width:18, height:18, borderRadius:"50%", background:col, border:color===col?`3px solid ${C.accent}`:`1px solid ${C.border}`, cursor:"pointer", flexShrink:0 }} />
          ))}
        </div>
        <div style={{ width:1, height:20, background:C.border }} />
        <div style={{ display:"flex", alignItems:"center", gap:5 }}>
          <span style={{ fontSize:11, color:C.muted }}>Size</span>
          <input type="range" min="1" max="20" value={brushSize} onChange={e => setBrushSize(+e.target.value)} style={{ width:60 }} />
        </div>
        <button onClick={clearDrawing} style={{ fontSize:12, background:"none", border:`1px solid ${C.border}`, borderRadius:6, padding:"4px 8px", cursor:"pointer", color:C.muted }}>🗑️</button>
        <div style={{ flex:1 }} />
        {/* Page nav */}
        {(isPDF || isPPT) && totalPages > 0 && (
          <div style={{ display:"flex", alignItems:"center", gap:5 }}>
            <button onClick={() => changePage(Math.max(1, page-1))} disabled={page===1}
              style={{ width:28, height:28, borderRadius:6, border:`1px solid ${C.border}`, background:"#fff", cursor:page===1?"not-allowed":"pointer", opacity:page===1?.4:1 }}>‹</button>
            <input type="number" min="1" max={totalPages} value={page}
              onChange={e => { const v=parseInt(e.target.value); if(v>=1&&v<=totalPages) changePage(v); }}
              style={{ width:40, textAlign:"center", border:`1px solid ${C.border}`, borderRadius:6, padding:"3px 2px", fontSize:13, outline:"none" }} />
            <span style={{ fontSize:12, color:C.muted }}>/{totalPages}</span>
            <button onClick={() => changePage(Math.min(totalPages, page+1))} disabled={page===totalPages}
              style={{ width:28, height:28, borderRadius:6, border:`1px solid ${C.border}`, background:"#fff", cursor:page===totalPages?"not-allowed":"pointer", opacity:page===totalPages?.4:1 }}>›</button>
          </div>
        )}
        {/* AI Explain */}
        {isPDF && totalPages > 0 ? (
          <div style={{ display:"flex", alignItems:"center", gap:5 }}>
            <span style={{ fontSize:11, color:C.muted }}>Explain pg</span>
            <input type="number" min="1" max={totalPages} value={explainPageNum}
              onChange={e => setExplainPageNum(Math.max(1, Math.min(totalPages, parseInt(e.target.value)||1)))}
              style={{ width:40, textAlign:"center", border:`1px solid ${C.border}`, borderRadius:6, padding:"3px 2px", fontSize:13, outline:"none" }} />
            <button onClick={() => doExplain(explainPageNum)} disabled={explaining}
              style={{ display:"flex", alignItems:"center", gap:5, background:C.accent, color:"#fff", border:"none", borderRadius:7, padding:"6px 12px", fontSize:12, fontWeight:600, cursor:explaining?"not-allowed":"pointer" }}>
              <Icon d={I.sparkle} size={12} color="#fff" sw={2} />{explaining?"…":"Explain"}
            </button>
          </div>
        ) : (isImage || isText || isOffice) && (
          <button onClick={() => doExplain(1)} disabled={explaining}
            style={{ display:"flex", alignItems:"center", gap:5, background:C.accent, color:"#fff", border:"none", borderRadius:7, padding:"6px 12px", fontSize:12, fontWeight:600, cursor:explaining?"not-allowed":"pointer" }}>
            <Icon d={I.sparkle} size={12} color="#fff" sw={2} />{explaining?"Explaining…":"AI Explain"}
          </button>
        )}
      </div>

      {/* Viewer area */}
      <div style={{ flex:1, display:"flex", overflow:"hidden" }}>
        <div style={{ flex:1, overflow:"auto", background:"#3a3a3a", display:"flex", justifyContent:"center", alignItems:"flex-start", padding:"20px", position:"relative" }}>
          {loading && (
            <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center", justifyContent:"center", background:"#3a3a3a" }}>
              <div style={{ textAlign:"center", color:"#fff" }}>
                <div style={{ fontSize:32, marginBottom:10 }}>📄</div>
                <p style={{ fontSize:14 }}>Loading file…</p>
              </div>
            </div>
          )}
          {/* PDF */}
          {isPDF && (
            <div style={{ position:"relative", boxShadow:"0 8px 40px rgba(0,0,0,.5)", display:"inline-block" }}>
              <canvas ref={canvasRef} style={{ display:"block", maxWidth:"100%" }} />
              <canvas ref={drawCanvasRef}
                style={{ position:"absolute", top:0, left:0, width:"100%", height:"100%", cursor:tool==="eraser"?"cell":"crosshair", touchAction:"none" }}
                onMouseDown={startDraw} onMouseMove={doDraw} onMouseUp={stopDraw} onMouseLeave={stopDraw}
                onTouchStart={startDraw} onTouchMove={doDraw} onTouchEnd={stopDraw} />
            </div>
          )}
          {isImage && imgSrc && <img src={imgSrc} alt={file.name} style={{ maxWidth:"100%", boxShadow:"0 8px 40px rgba(0,0,0,.5)" }} />}
          {isText && <TextViewer fileObj={fileObj} />}
          {isWord && <WordViewer fileObj={fileObj} />}
          {isPPT && <PPTViewer fileObj={fileObj} page={page} onTotalPages={setTotalPages} />}
          {isExcel && <ExcelViewer fileObj={fileObj} />}
          {!isPDF && !isImage && !isText && !isOffice && (
            <div style={{ textAlign:"center", color:"#fff", padding:"60px 20px" }}>
              <div style={{ fontSize:56, marginBottom:14 }}>📎</div>
              <p style={{ fontSize:15, fontWeight:600, marginBottom:6 }}>{file.name}</p>
              <p style={{ opacity:.6, fontSize:13 }}>Preview not available. Use AI Assistant tab.</p>
            </div>
          )}
        </div>
        {/* AI Explain panel */}
        {showExplain && (
          <div style={{ width:320, minWidth:260, background:C.surface, borderLeft:`1px solid ${C.border}`, display:"flex", flexDirection:"column", overflow:"hidden" }}>
            <div style={{ padding:"12px 16px", borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
              <div style={{ display:"flex", alignItems:"center", gap:7 }}>
                <Icon d={I.sparkle} size={14} color={C.accent} sw={2} />
                <span style={{ fontSize:13, fontWeight:700, color:C.text }}>{isPDF ? `Page ${explainPageNum}` : "AI"} Explanation</span>
              </div>
              <button onClick={() => setShowExplain(false)} style={{ background:"none", border:"none", cursor:"pointer" }}>
                <Icon d={I.x} size={14} color={C.muted} />
              </button>
            </div>
            <div style={{ flex:1, overflowY:"auto", padding:"14px 16px" }}>
              {explaining
                ? <div style={{ display:"flex", gap:5, paddingTop:8 }}>{[0,1,2].map(j=><div key={j} style={{ width:7,height:7,borderRadius:"50%",background:C.accent,animation:"bounce 1.2s infinite",animationDelay:`${j*.2}s` }}/>)}</div>
                : <div style={{ fontSize:13, lineHeight:1.7, color:C.text }}><Fmt text={explanation} /></div>
              }
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


function TextViewer({ fileObj }) {
  const [text, setText] = useState("");
  useEffect(() => {
    readFileAsText(fileObj).then(t => setText(t || "File appears to be empty.")).catch(() => setText("Could not read file."));
  }, [fileObj]);
  return (
    <div style={{ background:"#fff", padding:"24px 28px", borderRadius:4, maxWidth:800, width:"100%", boxShadow:"0 8px 32px rgba(0,0,0,.4)", whiteSpace:"pre-wrap", fontFamily:"monospace", fontSize:13, lineHeight:1.7, color:"#1a1a1a", minHeight:400, wordBreak:"break-word", overflowWrap:"break-word", overflowX:"hidden" }}>
      {text || "Loading…"}
    </div>
  );
}

function WordViewer({ fileObj }) {
  const [html, setHtml] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    (async () => {
      try {
        if (!window.mammoth) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.6.0/mammoth.browser.min.js";
            s.onload = res; s.onerror = rej;
            document.head.appendChild(s);
          });
        }
        const ab = await fileObj.arrayBuffer();
        const result = await window.mammoth.convertToHtml({ arrayBuffer: ab });
        setHtml(result.value);
      } catch(e) { setHtml(`<p style="color:red">Could not render: ${e.message}</p>`); }
      setLoading(false);
    })();
  }, [fileObj]);
  return (
    <div style={{ background:"#fff", padding:"40px 56px", borderRadius:4, maxWidth:850, width:"100%", boxShadow:"0 8px 32px rgba(0,0,0,.4)", minHeight:500, lineHeight:1.8, color:"#1a1a1a", fontSize:14 }}>
      {loading ? <p style={{ color:"#888" }}>Loading Word document…</p> : <div dangerouslySetInnerHTML={{ __html: html }} />}
    </div>
  );
}

function PPTViewer({ fileObj, page, onTotalPages }) {
  const [slides, setSlides] = useState([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    (async () => {
      try {
        if (!window.JSZip) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
            s.onload = res; s.onerror = rej;
            document.head.appendChild(s);
          });
        }
        const ab = await fileObj.arrayBuffer();
        const zip = await window.JSZip.loadAsync(ab);
        const slideFiles = Object.keys(zip.files)
          .filter(f => /^ppt[/\\]slides[/\\]slide[0-9]+\.xml$/.test(f))
          .sort((a,b) => parseInt(a.match(/\d+/)?.[0]||0) - parseInt(b.match(/\d+/)?.[0]||0));
        onTotalPages(slideFiles.length);
        const parsed = [];
        for (const sf of slideFiles) {
          const xml = await zip.files[sf].async("string");
          const allT = [...xml.matchAll(/<a:t>([^<]*)<\/a:t>/g)].map(m => m[1]).filter(t => t.trim());
          const bgMatch = xml.match(/solidFill[\s\S]*?<a:srgbClr val="([^"]+)"/);
          parsed.push({ texts: allT, bg: bgMatch ? "#" + bgMatch[1] : "#ffffff" });
        }
        setSlides(parsed);
      } catch(e) { setSlides([{ texts: [`Error loading: ${e.message}`], bg:"#fff" }]); }
      setLoading(false);
    })();
  }, [fileObj]);

  if (loading) return <div style={{ color:"#fff", fontSize:15 }}>Loading presentation…</div>;
  const slide = slides[Math.min(page-1, slides.length-1)];
  if (!slide) return null;
  const isDark = slide.bg !== "#ffffff" && slide.bg !== "#fff" && slide.bg !== "#FFFFFF";
  return (
    <div style={{ width:720, minHeight:405, background:slide.bg, borderRadius:8, boxShadow:"0 8px 40px rgba(0,0,0,.5)", padding:"48px 56px", display:"flex", flexDirection:"column", gap:14 }}>
      {slide.texts.length === 0
        ? <p style={{ color:"#aaa", textAlign:"center", margin:"auto", fontSize:14 }}>Empty slide</p>
        : slide.texts.map((t,i) => (
          <p key={i} style={{ fontSize:i===0?26:15, fontWeight:i===0?700:400, color:isDark?"#fff":"#1a1a1a", lineHeight:1.5, fontFamily:"'DM Sans',sans-serif",
            borderBottom:i===0&&slide.texts.length>1?"1px solid rgba(128,128,128,.2)":"none", paddingBottom:i===0&&slide.texts.length>1?16:0 }}>
            {i>0?"• ":""}{t}
          </p>
        ))
      }
    </div>
  );
}

function ExcelViewer({ fileObj }) {
  const [sheets, setSheets] = useState([]);
  const [activeSheet, setActiveSheet] = useState(0);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    (async () => {
      try {
        if (!window.XLSX) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js";
            s.onload = res; s.onerror = rej;
            document.head.appendChild(s);
          });
        }
        const ab = await fileObj.arrayBuffer();
        const wb = window.XLSX.read(ab, { type:"array" });
        const parsed = wb.SheetNames.map(name => ({
          name,
          html: window.XLSX.utils.sheet_to_html(wb.Sheets[name], { editable:false }),
        }));
        setSheets(parsed);
      } catch(e) { setSheets([{ name:"Error", html:`<p style="color:red">${e.message}</p>` }]); }
      setLoading(false);
    })();
  }, [fileObj]);
  return (
    <div style={{ background:"#fff", borderRadius:4, maxWidth:"95%", width:"100%", boxShadow:"0 8px 32px rgba(0,0,0,.4)", overflow:"hidden" }}>
      {sheets.length > 1 && (
        <div style={{ display:"flex", borderBottom:"1.5px solid #e2e8f0", background:"#f7f7f7" }}>
          {sheets.map((s,i) => (
            <button key={i} onClick={() => setActiveSheet(i)}
              style={{ padding:"10px 20px", border:"none", borderBottom:activeSheet===i?"2px solid #3D5A80":"2px solid transparent", background:"none", cursor:"pointer", fontSize:13, fontWeight:activeSheet===i?700:400, color:activeSheet===i?"#3D5A80":"#666" }}>
              {s.name}
            </button>
          ))}
        </div>
      )}
      <div style={{ overflow:"auto", maxHeight:"calc(100vh - 280px)", padding:16 }}>
        {loading ? <p style={{ color:"#666", padding:24 }}>Loading spreadsheet…</p>
          : sheets[activeSheet] && <div dangerouslySetInnerHTML={{ __html: sheets[activeSheet].html }} />}
      </div>
      <style>{`table{border-collapse:collapse;width:100%}td,th{border:1px solid #e2e8f0;padding:6px 10px;min-width:80px;white-space:nowrap}th{background:#f0f4f8;font-weight:600}`}</style>
    </div>
  );
}


// ─── NOTES TAB ────────────────────────────────────────────────────────────────
// ─── AI TAB ───────────────────────────────────────────────────────────────────
function AITab({ file, allFiles, folder, onUpdate }) {
  const [msgs, setMsgs] = useState([]);
  const [inp, setInp] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);

  const send = async () => {
    const text = inp.trim();
    if (!text || loading) return;
    const userMsg = { role:"user", content: text };
    const newMsgs = [...msgs, userMsg];
    setMsgs(newMsgs); setInp(""); setLoading(true);
    try {
      const fileContext = file ? `You are helping a student with the file "${file.name}".` : `You are helping a student with the folder "${folder?.name}".`;
      const reply = await callClaudeChat(
        fileContext + " Answer clearly and helpfully. Use bullet points and bold for key info.",
        newMsgs.map(m => ({ role: m.role, content: m.content }))
      );
      setMsgs([...newMsgs, { role:"assistant", content: reply }]);
    } catch(e) { setMsgs([...newMsgs, { role:"assistant", content:"Error: " + e.message }]); }
    setLoading(false);
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"calc(100vh - 200px)", minHeight:400 }}>
      <div style={{ flex:1, overflowY:"auto", padding:"16px 0", display:"flex", flexDirection:"column", gap:12 }}>
        {msgs.length === 0 && (
          <div style={{ textAlign:"center", padding:"40px 20px", color:C.muted }}>
            <div style={{ fontSize:40, marginBottom:12 }}>🤖</div>
            <p style={{ fontSize:15, fontWeight:600, color:C.text, marginBottom:6 }}>AI Assistant</p>
            <p style={{ fontSize:13 }}>Ask anything about {file ? `"${file.name}"` : "your files"}.</p>
          </div>
        )}
        {msgs.map((m,i) => (
          <div key={i} style={{ display:"flex", justifyContent:m.role==="user"?"flex-end":"flex-start" }}>
            <div style={{ maxWidth:"80%", padding:"10px 14px", borderRadius:14, background:m.role==="user"?C.accent:C.surface, color:m.role==="user"?"#fff":C.text, fontSize:14, lineHeight:1.6, border:m.role==="user"?"none":`1px solid ${C.border}` }}>
              <Fmt text={m.content} />
            </div>
          </div>
        ))}
        {loading && <div style={{ display:"flex", gap:5, padding:"10px 14px" }}>{[0,1,2].map(j=><div key={j} style={{ width:7,height:7,borderRadius:"50%",background:C.accent,animation:"bounce 1.2s infinite",animationDelay:`${j*.2}s` }}/>)}</div>}
        <div ref={bottomRef} />
      </div>
      <div style={{ display:"flex", gap:10, paddingTop:12, borderTop:`1px solid ${C.border}` }}>
        <input value={inp} onChange={e=>setInp(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}}
          placeholder="Ask a question…"
          style={{ flex:1, border:`1.5px solid ${C.border}`, borderRadius:12, padding:"11px 16px", fontSize:14, outline:"none", background:C.bg, color:C.text }} />
        <button onClick={send} disabled={!inp.trim()||loading}
          style={{ background:inp.trim()&&!loading?C.accent:"#ccc", color:"#fff", border:"none", borderRadius:12, padding:"11px 20px", fontSize:14, fontWeight:600, cursor:inp.trim()&&!loading?"pointer":"not-allowed" }}>
          Send
        </button>
      </div>
    </div>
  );
}

function NotesTab({ file, onUpdate, user, isGuest }) {
  const [notes, setNotes] = useState(file.notes||"");
  const [gen, setGen] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showTopicInput, setShowTopicInput] = useState(false);
  const [customTopic, setCustomTopic] = useState("");
  const [recording, setRecording] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("");
  const recognitionRef = useRef(null);
  const transcriptRef = useRef("");

  const generate = async () => {
    setGen(true);
    try {
      const fileObj = file._fileObj || FILE_STORE.get(file.id) || null;
      const fileText = fileObj ? await extractFileText(fileObj) : null;

      const userMsg = fileText
        ? `Here is the content from the file "${file.name}":\n\n${fileText.slice(0, 6000)}\n\nNow create detailed study notes based ONLY on this content.`
        : `Create comprehensive study notes for a subject/topic named "${file.name}". Make them detailed and useful for exam revision.`;

      const styleInstructions = {
        detailed: "Write detailed comprehensive notes with headings, subpoints, definitions and examples. Aim for 25-35 lines.",
        bullet: "Write ONLY bullet points. No paragraphs. Every fact on its own bullet line. Group under short bold headings.",
        simple: "Write very simple short notes. Use plain language like explaining to a 14-year-old. Short sentences. No jargon.",
        exam: "Write exam-focused notes. Include: likely exam questions, key terms to memorise, formulas, dates, and a quick revision checklist at the end.",
      };
      const txt = await callClaude(
        `You are an expert study notes writer. ${styleInstructions[noteStyle] || styleInstructions.detailed}
Rules:
- Do NOT use asterisks (*) anywhere in your output
- Use UPPERCASE HEADINGS instead of bold markdown
- Use • for bullet points
- Write clean readable text with no markdown symbols
- ONLY use information from the provided file content if given`,
        userMsg
      );
      setNotes(txt); onUpdate({...file,notes:txt});
    } catch(e){ setNotes(`Error: ${e.message}`); }
    setGen(false);
  };

  const save = () => { onUpdate({...file,notes}); setSaved(true); setTimeout(()=>setSaved(false),2000); };

  const isRecordingRef = useRef(false);

  const startVoice = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { setVoiceStatus("Speech recognition is not supported. Please use Chrome or Edge."); return; }
    transcriptRef.current = "";
    isRecordingRef.current = true;
    const recognition = new SR();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;
    recognition.onresult = (e) => {
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) {
          transcriptRef.current += e.results[i][0].transcript + " ";
        } else {
          interim += e.results[i][0].transcript;
        }
      }
      const display = transcriptRef.current + (interim ? interim : "");
      if (display.trim()) setVoiceStatus("🎙️ Hearing: " + display.slice(-60));
    };
    recognition.onerror = (e) => {
      if (e.error === "not-allowed") setVoiceStatus("Microphone access denied. Please allow mic access.");
      else if (e.error !== "no-speech") setVoiceStatus("Mic error: " + e.error);
    };
    recognition.onend = () => {
      if (isRecordingRef.current) {
        try { recognition.start(); } catch(err) {}
      }
    };
    recognition.start();
    recognitionRef.current = recognition;
    setRecording(true);
    setVoiceStatus("🎙️ Recording… speak now. Click Stop when done.");
  };

  const stopVoice = async () => {
    isRecordingRef.current = false;
    setRecording(false);
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      try { recognitionRef.current.stop(); } catch {}
      recognitionRef.current = null;
    }
    await new Promise(r => setTimeout(r, 300));
    const raw = transcriptRef.current.trim();
    if (!raw) { setVoiceStatus("Nothing was recorded. Make sure your microphone is working and try again."); setTimeout(() => setVoiceStatus(""), 4000); return; }
    setProcessing(true);
    setVoiceStatus("✨ Organising your notes…");
    try {
      const result = await callClaude(
        `You are an expert note-taker. A student recorded a voice note from a lecture or study session.
Rules:
1. ONLY use what was said — never add outside information
2. Fix grammar, remove filler words (um, uh, like, you know)
3. Organise into clear notes with **BOLD HEADINGS**
4. Use bullet points (•) for key facts
5. Bold important terms
6. Make notes clear and useful for exam revision
7. Do your best even if the recording is short or unclear`,
        `Turn this raw speech into clean organised study notes:

"${raw}"`
      );
      const combined = notes ? notes + "\n\n---\n\n" + result : result;
      setNotes(combined); onUpdate({...file, notes: combined});
      setVoiceStatus("✅ Notes added!");
      setTimeout(() => setVoiceStatus(""), 3000);
    } catch(e) { setVoiceStatus("Error: " + e.message); }
    setProcessing(false);
  };

  const generateWithTopic = async (topic) => {
    setGen(true);
    setShowTopicInput(false);
    try {
      const fileObj = file._fileObj || FILE_STORE.get(file.id) || null;
      const fileText = fileObj ? await extractFileText(fileObj) : null;

      const userMsg = fileText
        ? `Here is the content from the file "${file.name}":\n\n${fileText.slice(0, 6000)}\n\nNow create detailed study notes specifically about "${topic}" based on this content.`
        : `Create comprehensive study notes specifically about: "${topic}". Make them detailed and useful for exam revision.`;

      const txt = await callClaude(
        `You are an expert study notes writer. Create the BEST possible study notes a student can use to revise. Follow these rules:
1. Start with a 1-2 sentence summary of the topic
2. Break content into clear sections with **BOLD HEADINGS**
3. Use bullet points (•) for key facts under each heading
4. Highlight important terms in **bold**
5. Include formulas, definitions, or key dates if present
6. End with a "**Key Takeaways**" section with the 3-5 most important points
7. Write in simple clear language
8. Be thorough — aim for 20-30 lines of useful content
9. ONLY use information from the provided file content if given — do not add outside information`,
        userMsg
      );
      setNotes(txt); onUpdate({...file,notes:txt});
    } catch(e){ setNotes(`Error: ${e.message}`); }
    setGen(false);
  };

  const [noteStyle, setNoteStyle] = useState("detailed");
  const NOTE_STYLES = [
    {id:"detailed", label:"Detailed"},
    {id:"bullet", label:"Bullet Points"},
    {id:"simple", label:"Simple"},
    {id:"exam", label:"Exam Focused"},
  ];

  return (
    <div>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:16, flexWrap:"wrap", gap:10 }}>
        <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:22, fontWeight:700, color:C.text }}>Notes</h2>
        <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
          {/* Voice Notes — Google login only */}
          {!isGuest && user && (
            recording
              ? <button onClick={stopVoice} disabled={processing}
                  style={{ display:"flex", alignItems:"center", gap:7, background:"#FED7D7", color:C.red, border:`1.5px solid ${C.red}44`, borderRadius:10, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:"pointer", animation:"pulse 1.5s infinite" }}>
                  <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}`}</style>
                  ⏹️ Stop Recording
                </button>
              : <button onClick={startVoice} disabled={gen || processing} className="hov"
                  style={{ display:"flex", alignItems:"center", gap:7, background:"#FFF5F5", color:C.red, border:`1.5px solid ${C.red}44`, borderRadius:10, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:"pointer" }}>
                  🎙️ Voice Notes
                </button>
          )}
          <button onClick={generate} disabled={gen} className="hov"
            style={{ display:"flex", alignItems:"center", gap:7, background:C.accentL, color:C.accent, border:"none", borderRadius:10, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:gen?"not-allowed":"pointer" }}>
            <Icon d={gen?I.refresh:I.sparkle} size={15} color={C.accent} />{gen?"Generating…":"AI Generate"}
          </button>
          <button onClick={() => setShowTopicInput(true)} disabled={gen} className="hov"
            style={{ display:"flex", alignItems:"center", gap:7, background:C.surface, color:C.text, border:`1.5px solid ${C.border}`, borderRadius:10, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:gen?"not-allowed":"pointer" }}>
            <Icon d={I.edit} size={14} color={C.text} /> Custom Topic
          </button>
          <button onClick={save} className="hov"
            style={{ display:"flex", alignItems:"center", gap:7, background:saved?C.greenL:C.accent, color:saved?C.green:"#fff", border:"none", borderRadius:10, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:"pointer" }}>
            <Icon d={saved?I.check:I.edit} size={14} color={saved?C.green:"#fff"} />{saved?"Saved!":"Save"}
          </button>
        </div>
      </div>
      {/* Note style selector */}
      <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:12, flexWrap:"wrap" }}>
        <span style={{ fontSize:12, color:C.muted, fontWeight:600 }}>Style:</span>
        {NOTE_STYLES.map(s => (
          <button key={s.id} onClick={() => setNoteStyle(s.id)}
            style={{ fontSize:12, padding:"4px 10px", borderRadius:20, border:`1.5px solid ${noteStyle===s.id?C.accent:C.border}`, background:noteStyle===s.id?C.accentL:"#fff", color:noteStyle===s.id?C.accent:C.muted, cursor:"pointer", fontWeight:noteStyle===s.id?700:400 }}>
            {s.label}
          </button>
        ))}
      </div>
      {voiceStatus && (
        <div style={{ background: voiceStatus.startsWith("✅") ? C.greenL : voiceStatus.startsWith("🎙️") ? "#FFF5F5" : voiceStatus.startsWith("✨") ? C.accentL : C.redL,
          border:`1.5px solid ${voiceStatus.startsWith("✅") ? C.green : voiceStatus.startsWith("🎙️") ? C.red : voiceStatus.startsWith("✨") ? C.accentS : C.red}44`,
          borderRadius:10, padding:"10px 16px", marginBottom:14, fontSize:14, fontWeight:500,
          color: voiceStatus.startsWith("✅") ? C.green : voiceStatus.startsWith("🎙️") ? C.red : C.text }}>
          {voiceStatus}
          {!isGuest && !user && <span style={{ marginLeft:8, fontSize:12, color:C.muted }}>— Sign in with Google to use voice notes</span>}
        </div>
      )}
      {isGuest && (
        <div style={{ background:C.warmL, border:`1.5px solid ${C.warm}33`, borderRadius:10, padding:"10px 16px", marginBottom:14, fontSize:13, color:C.warm, fontWeight:500 }}>
          🎙️ Voice Notes is available for Google account users only — <strong>sign in with Google</strong> to unlock it
        </div>
      )}

      {showTopicInput && (
        <div style={{ background:C.accentL, border:`1.5px solid ${C.accentS}`, borderRadius:12, padding:16, marginBottom:16, display:"flex", gap:10, alignItems:"center" }}>
          <input autoFocus value={customTopic} onChange={e => setCustomTopic(e.target.value)}
            onKeyDown={e => { if(e.key==="Enter" && customTopic.trim()) generateWithTopic(customTopic.trim()); }}
            placeholder="e.g. Photosynthesis, World War 2, Quadratic equations…"
            style={{ flex:1, border:`1.5px solid ${C.accentS}`, borderRadius:8, padding:"9px 12px", fontSize:14, outline:"none", color:C.text, background:C.surface }} />
          <button onClick={() => customTopic.trim() && generateWithTopic(customTopic.trim())}
            disabled={!customTopic.trim()}
            style={{ background:C.accent, color:"#fff", border:"none", borderRadius:8, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:customTopic.trim()?"pointer":"not-allowed" }}>
            Generate
          </button>
          <button onClick={() => setShowTopicInput(false)}
            style={{ background:"none", border:"none", cursor:"pointer", color:C.muted, padding:"9px 6px" }}>✕</button>
        </div>
      )}

      <textarea value={notes} onChange={e=>setNotes(e.target.value)}
        placeholder="Write notes here, or click 'AI Generate' to create them automatically…"
        style={{ width:"100%", minHeight:420, border:`1.5px solid ${C.border}`, borderRadius:14, padding:"20px", fontSize:15, lineHeight:1.7, outline:"none", resize:"vertical", color:C.text, background:C.surface, fontFamily:"'DM Sans',sans-serif" }} />
    </div>
  );
}

// ─── STUDY CARDS TAB ──────────────────────────────────────────────────────────
function CardsTab({ file, onUpdate }) {
  const [cards, setCards] = useState(file.studyCards||[]);
  const [flipped, setFlipped] = useState({});
  const [gen, setGen] = useState(false);
  const [showAdd, setShowAdd] = useState(false);
  const [showCountPicker, setShowCountPicker] = useState(false);
  const [cardCount, setCardCount] = useState(8);
  const [nQ, setNQ] = useState(""); const [nA, setNA] = useState("");

  const generate = async (count = cardCount) => {
    setGen(true);
    setShowCountPicker(false);
    try {
      const fileObj = file._fileObj || FILE_STORE.get(file.id) || null;
      const fileText = fileObj ? await extractFileText(fileObj) : null;
      const userMsg = fileText
        ? `Here is the content from "${file.name}":\n\n${fileText.slice(0, 5000)}\n\nCreate exactly ${count} study flashcards based ONLY on this content. Return JSON array: [{"question":"…","answer":"…"}]`
        : `Create exactly ${count} study flashcards for the topic "${file.name}". Return JSON array: [{"question":"…","answer":"…"}]`;
      const txt = await callClaude("Return ONLY valid JSON array. No markdown, no explanation, no extra text.", userMsg);
      const parsed = JSON.parse(txt.replace(/```json|```/g,"").trim());
      const nc = parsed.map((c,i)=>({id:Date.now()+i,...c}));
      setCards(nc); onUpdate({...file,studyCards:nc});
    } catch(e){ console.error(e); }
    setGen(false);
  };

  const del = (id) => { const u=cards.filter(c=>c.id!==id); setCards(u); onUpdate({...file,studyCards:u}); };
  const add = () => {
    if(!nQ.trim()||!nA.trim()) return;
    const u=[...cards,{id:Date.now(),question:nQ,answer:nA}];
    setCards(u); onUpdate({...file,studyCards:u}); setNQ(""); setNA(""); setShowAdd(false);
  };

  return (
    <div>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:20 }}>
        <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:22, fontWeight:700, color:C.text }}>Study Cards <span style={{ fontSize:15, fontWeight:500, color:C.muted }}>({cards.length})</span></h2>
        <div style={{ display:"flex", gap:10 }}>
          <button onClick={() => setShowAdd(true)} className="hov"
            style={{ display:"flex", alignItems:"center", gap:7, background:C.surface, color:C.text, border:`1.5px solid ${C.border}`, borderRadius:10, padding:"9px 14px", fontSize:14, fontWeight:600, cursor:"pointer" }}>
            <Icon d={I.plus} size={14} color={C.text} sw={2.5} /> Add Card
          </button>
          <button onClick={() => setShowCountPicker(p => !p)} disabled={gen} className="hov"
            style={{ display:"flex", alignItems:"center", gap:7, background:C.accentL, color:C.accent, border:"none", borderRadius:10, padding:"9px 16px", fontSize:14, fontWeight:600, cursor:gen?"not-allowed":"pointer" }}>
            <Icon d={gen?I.refresh:I.sparkle} size={15} color={C.accent} />{gen?"Generating…":"AI Generate"}
          </button>
        </div>
      </div>

      {showCountPicker && (
        <div style={{ background:C.accentL, border:`1.5px solid ${C.accentS}`, borderRadius:12, padding:16, marginBottom:16 }}>
          <p style={{ fontSize:13, fontWeight:600, color:C.accent, marginBottom:12 }}>How many cards do you want?</p>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:14 }}>
            {[5, 8, 10, 15, 20, 25, 30].map(n => (
              <button key={n} onClick={() => setCardCount(n)}
                style={{ width:48, height:40, borderRadius:8, border:`1.5px solid ${cardCount===n?C.accent:C.border}`, background:cardCount===n?C.accent:"#fff", color:cardCount===n?"#fff":C.text, fontSize:14, fontWeight:600, cursor:"pointer" }}>
                {n}
              </button>
            ))}
          </div>
          <div style={{ display:"flex", gap:10, alignItems:"center" }}>
            <div style={{ display:"flex", alignItems:"center", gap:8, flex:1 }}>
              <span style={{ fontSize:13, color:C.muted }}>Custom:</span>
              <input type="number" min="0" max="50" value={cardCount} onChange={e => setCardCount(Math.min(50, Math.max(0, parseInt(e.target.value)||0)))}
                style={{ width:70, border:`1.5px solid ${C.border}`, borderRadius:8, padding:"7px 10px", fontSize:14, outline:"none", color:C.text, background:"#fff" }} />
            </div>
            <button onClick={() => generate(cardCount)} disabled={gen}
              style={{ background:C.accent, color:"#fff", border:"none", borderRadius:8, padding:"9px 20px", fontSize:14, fontWeight:600, cursor:"pointer" }}>
              Generate {cardCount} Cards
            </button>
            <button onClick={() => setShowCountPicker(false)}
              style={{ background:"none", border:"none", cursor:"pointer", color:C.muted, padding:"9px 6px" }}>✕</button>
          </div>
        </div>
      )}
      {showAdd && (
        <div style={{ background:C.surface, border:`1.5px solid ${C.accentS}`, borderRadius:14, padding:20, marginBottom:20 }}>
          <input value={nQ} onChange={e=>setNQ(e.target.value)} placeholder="Question" style={{ width:"100%", border:`1.5px solid ${C.border}`, borderRadius:8, padding:"9px 12px", fontSize:14, marginBottom:10, outline:"none", color:C.text, background:C.bg }} />
          <input value={nA} onChange={e=>setNA(e.target.value)} placeholder="Answer" style={{ width:"100%", border:`1.5px solid ${C.border}`, borderRadius:8, padding:"9px 12px", fontSize:14, marginBottom:14, outline:"none", color:C.text, background:C.bg }} />
          <div style={{ display:"flex", gap:8 }}>
            <button onClick={()=>setShowAdd(false)} style={{ flex:1, padding:"8px", border:`1.5px solid ${C.border}`, borderRadius:8, background:"none", cursor:"pointer", fontSize:14, color:C.text }}>Cancel</button>
            <button onClick={add} style={{ flex:2, padding:"8px", background:C.accent, color:"#fff", border:"none", borderRadius:8, cursor:"pointer", fontSize:14, fontWeight:600 }}>Add</button>
          </div>
        </div>
      )}
      {cards.length === 0
        ? <div style={{ textAlign:"center", padding:"60px 0", color:C.muted }}><Icon d={I.cards} size={40} color={C.border} /><p style={{ marginTop:12, fontSize:15 }}>No cards yet — generate or add some</p></div>
        : <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(260px,1fr))", gap:16 }}>
            {cards.map(card => (
              <div key={card.id} onClick={() => setFlipped(f=>({...f,[card.id]:!f[card.id]}))}
                style={{ background:flipped[card.id]?C.accentL:C.surface, border:`1.5px solid ${flipped[card.id]?C.accentS:C.border}`, borderRadius:16, padding:24, cursor:"pointer", minHeight:140, display:"flex", flexDirection:"column", justifyContent:"space-between", transition:"all .2s" }}>
                <div>
                  <p style={{ fontSize:11, fontWeight:700, color:flipped[card.id]?C.accent:C.muted, letterSpacing:1, marginBottom:10, textTransform:"uppercase" }}>{flipped[card.id]?"Answer":"Question"}</p>
                  <p style={{ fontSize:15, color:C.text, lineHeight:1.5 }}>{flipped[card.id]?card.answer:card.question}</p>
                </div>
                <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginTop:16 }}>
                  <span style={{ fontSize:12, color:C.muted }}>Tap to flip</span>
                  <button onClick={e=>{e.stopPropagation();del(card.id);}} style={{ background:"none", border:"none", cursor:"pointer", padding:2 }}>
                    <Icon d={I.trash} size={14} color={C.muted} />
                  </button>
                </div>
              </div>
            ))}
          </div>
      }
    </div>
  );
}

// ─── GAME TAB ─────────────────────────────────────────────────────────────────
function GameTab({ file }) {
  const cards = file.studyCards || [];
  const [activeGame, setActiveGame] = useState(null);

  if (cards.length === 0) return (
    <div style={{ textAlign:"center", padding:"80px 0" }}>
      <Icon d={I.game} size={48} color={C.border} />
      <p style={{ fontSize:18, fontWeight:600, color:C.text, marginTop:16, marginBottom:8 }}>No cards yet</p>
      <p style={{ fontSize:14, color:C.muted }}>Generate study cards first, then come back to play</p>
    </div>
  );

  if (activeGame==="mcq") return <MCQ cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="scramble") return <Scramble cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="match") return <Match cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="falling") return <Falling cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="tower") return <Tower cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="speedrun") return <Speedrun cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="truefalse") return <TrueFalse cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="memory") return <Memory cards={cards} onBack={()=>setActiveGame(null)} />;

  const GAMES = [
    {id:"mcq",emoji:"🧠",title:"Multiple Choice",desc:"Pick the correct answer from 4 options",bg:C.accentL,accent:C.accent},
    {id:"scramble",emoji:"🔀",title:"Word Scramble",desc:"Unscramble the answer letters",bg:C.warmL,accent:C.warm},
    {id:"match",emoji:"🔗",title:"Matching Pairs",desc:"Connect each term to its definition",bg:C.greenL,accent:C.green},
    {id:"falling",emoji:"🧱",title:"Falling Blocks",desc:"Type the answer before the block falls!",bg:C.purpleL,accent:C.purple},
    {id:"tower",emoji:"🏗️",title:"Answer Tower",desc:"Build a tower — answer correctly to stack blocks",bg:"#E6FFFA",accent:"#2C7A7B"},
    {id:"speedrun",emoji:"⚡",title:"Speed Run",desc:"Answer as many as you can in 60 seconds!",bg:"#FFFFF0",accent:"#D69E2E"},
    {id:"truefalse",emoji:"✅",title:"True or False",desc:"Decide if the statement is true or false",bg:"#FAF5FF",accent:"#6B46C1"},
    {id:"memory",emoji:"🃏",title:"Memory Flip",desc:"Flip cards and match question to answer",bg:"#FFF5F5",accent:"#C53030"},
  ];

  return (
    <div style={{ maxWidth:760, margin:"0 auto" }}>
      <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:26, fontWeight:700, color:C.text, marginBottom:6 }}>Game Mode</h2>
      <p style={{ fontSize:14, color:C.muted, marginBottom:28 }}>{cards.length} cards ready · Choose a game</p>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))", gap:14 }}>
        {GAMES.map(g => (
          <button key={g.id} onClick={()=>setActiveGame(g.id)}
            style={{ background:g.bg, border:`1.5px solid ${g.accent}22`, borderRadius:18, padding:"20px 18px", textAlign:"left", cursor:"pointer", transition:"transform .15s,box-shadow .15s" }}
            onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-3px)";e.currentTarget.style.boxShadow=`0 8px 24px ${g.accent}33`;}}
            onMouseLeave={e=>{e.currentTarget.style.transform="none";e.currentTarget.style.boxShadow="none";}}>
            <div style={{ fontSize:32, marginBottom:10 }}>{g.emoji}</div>
            <p style={{ fontSize:15, fontWeight:700, color:C.text, marginBottom:5 }}>{g.title}</p>
            <p style={{ fontSize:12, color:C.muted, lineHeight:1.4 }}>{g.desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── GAMES ────────────────────────────────────────────────────────────────────
function GHeader({ title, score, curr, total, onBack, accent }) {
  return (
    <div style={{ marginBottom:20 }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:10 }}>
        <button onClick={onBack} style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14, padding:0 }}>
          <Icon d={I.back} size={16} color={C.muted} /> All Games
        </button>
        <span style={{ fontSize:14, fontWeight:700, color:accent }}>Score: {score}</span>
      </div>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:8 }}>
        <h3 style={{ fontFamily:"'Fraunces',serif", fontSize:20, fontWeight:700, color:C.text }}>{title}</h3>
        <span style={{ fontSize:13, color:C.muted }}>{curr+1}/{total}</span>
      </div>
      <div style={{ height:5, background:C.border, borderRadius:3 }}>
        <div style={{ height:"100%", width:`${(curr/total)*100}%`, background:accent, borderRadius:3, transition:"width .3s" }} />
      </div>
    </div>
  );
}

function GResults({ score, total, onBack, msg }) {
  const pct = Math.round(score/total*100);
  return (
    <div style={{ maxWidth:420, margin:"0 auto", textAlign:"center", padding:"40px 0" }}>
      <div style={{ fontSize:56, marginBottom:16 }}>{pct>=80?"🎉":pct>=50?"👍":"📚"}</div>
      <h2 style={{ fontFamily:"'Fraunces',serif", fontSize:28, fontWeight:700, color:C.text, marginBottom:6 }}>{msg||"Round Complete!"}</h2>
      <p style={{ fontSize:48, fontWeight:700, color:C.accent, marginBottom:4 }}>{score}/{total}</p>
      <p style={{ fontSize:16, color:C.muted, marginBottom:8 }}>{pct}% correct</p>
      <p style={{ fontSize:14, color:pct>=80?C.green:pct>=50?C.warm:C.red, fontWeight:600, marginBottom:32 }}>
        {pct>=80?"Excellent! 🌟":pct>=50?"Good effort! Keep studying.":"Keep practicing — you've got this!"}
      </p>
      <button onClick={onBack} style={{ background:C.accent, color:"#fff", border:"none", borderRadius:14, padding:"13px 32px", fontSize:15, fontWeight:700, cursor:"pointer" }}>← Back to Games</button>
    </div>
  );
}

function MCQ({ cards, onBack }) {
  const [deck]=useState(()=>[...cards].sort(()=>Math.random()-.5));
  const [curr,setCurr]=useState(0); const [sel,setSel]=useState(null); const [score,setScore]=useState(0); const [done,setDone]=useState(false);
  const mkOpts=(c)=>[...cards.filter(x=>x.id!==c.id).sort(()=>Math.random()-.5).slice(0,3).map(x=>x.answer),c.answer].sort(()=>Math.random()-.5);
  const [opts]=useState(()=>deck.map(mkOpts));
  const pick=(o)=>{ if(sel)return; setSel(o); if(o===deck[curr].answer)setScore(s=>s+1); };
  const next=()=>{ if(curr+1>=deck.length){setDone(true);return;} setCurr(c=>c+1); setSel(null); };
  if(done) return <GResults score={score} total={deck.length} onBack={onBack} />;
  return (
    <div style={{ maxWidth:560, margin:"0 auto" }}>
      <GHeader title="Multiple Choice" score={score} curr={curr} total={deck.length} onBack={onBack} accent={C.accent} />
      <div style={{ background:C.surface, border:`1.5px solid ${C.border}`, borderRadius:20, padding:"28px", marginBottom:16 }}>
        <p style={{ fontSize:12, fontWeight:700, color:C.muted, letterSpacing:1, textTransform:"uppercase", marginBottom:12 }}>Question</p>
        <p style={{ fontSize:17, color:C.text, lineHeight:1.6 }}>{deck[curr].question}</p>
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:10 }}>
        {opts[curr].map((o,i)=>{ const ok=o===deck[curr].answer,is=o===sel; let bg=C.surface,bd=C.border,col=C.text; if(sel){if(ok){bg=C.greenL;bd=C.green;col=C.green;}else if(is){bg=C.redL;bd=C.red;col=C.red;}} return <button key={i} onClick={()=>pick(o)} style={{ background:bg,border:`1.5px solid ${bd}`,borderRadius:12,padding:"14px 18px",textAlign:"left",fontSize:15,color:col,cursor:sel?"default":"pointer",fontWeight:is||(sel&&ok)?600:400,transition:"all .2s" }}><span style={{ fontWeight:700, marginRight:10, color:C.muted }}>{"ABCD"[i]}.</span>{o}</button>;})}
      </div>
      {sel && <button onClick={next} style={{ marginTop:16, width:"100%", background:C.accent, color:"#fff", border:"none", borderRadius:12, padding:"13px", fontSize:15, fontWeight:700, cursor:"pointer" }}>{curr+1>=deck.length?"See Results":"Next →"}</button>}
    </div>
  );
}

function Scramble({ cards, onBack }) {
  const [deck]=useState(()=>[...cards].sort(()=>Math.random()-.5).slice(0,Math.min(cards.length,8)));
  const [curr,setCurr]=useState(0);const [inp,setInp]=useState("");const [res,setRes]=useState(null);const [score,setScore]=useState(0);const [done,setDone]=useState(false);
  const sc=(s)=>{let x=s.split("").sort(()=>Math.random()-.5).join(""),t=0;while(x===s&&t++<10)x=s.split("").sort(()=>Math.random()-.5).join("");return x;};
  const [scr]=useState(()=>deck.map(c=>sc(c.answer)));
  const check=()=>{ const ok=inp.trim().toLowerCase()===deck[curr].answer.trim().toLowerCase(); setRes(ok?"correct":"wrong"); if(ok)setScore(s=>s+1); };
  const next=()=>{ if(curr+1>=deck.length){setDone(true);return;} setCurr(c=>c+1);setInp("");setRes(null); };
  if(done) return <GResults score={score} total={deck.length} onBack={onBack} />;
  return (
    <div style={{ maxWidth:500, margin:"0 auto" }}>
      <GHeader title="Word Scramble" score={score} curr={curr} total={deck.length} onBack={onBack} accent={C.warm} />
      <div style={{ background:C.surface, border:`1.5px solid ${C.border}`, borderRadius:20, padding:"28px", marginBottom:16 }}>
        <p style={{ fontSize:12, fontWeight:700, color:C.muted, letterSpacing:1, textTransform:"uppercase", marginBottom:10 }}>Question</p>
        <p style={{ fontSize:16, color:C.text, lineHeight:1.5, marginBottom:20 }}>{deck[curr].question}</p>
        <div style={{ display:"flex", flexWrap:"wrap", gap:6, marginBottom:20 }}>
          {scr[curr].split("").map((ch,i)=><div key={i} style={{ width:36,height:40,background:C.warmL,border:`1.5px solid ${C.warm}44`,borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",fontSize:18,fontWeight:700,color:C.warm }}>{ch===" "?"·":ch}</div>)}
        </div>
        <input value={inp} onChange={e=>{if(!res)setInp(e.target.value);}} onKeyDown={e=>{if(e.key==="Enter"&&!res&&inp.trim())check();}} placeholder="Type your answer…"
          style={{ width:"100%", border:`1.5px solid ${res==="correct"?C.green:res==="wrong"?C.red:C.border}`, borderRadius:10, padding:"11px 14px", fontSize:15, outline:"none", color:C.text, background:res==="correct"?C.greenL:res==="wrong"?C.redL:C.bg }} />
        {res&&<p style={{ marginTop:8, fontSize:14, fontWeight:600, color:res==="correct"?C.green:C.red }}>{res==="correct"?"✓ Correct!":"✗ Answer: "+deck[curr].answer}</p>}
      </div>
      {!res?<button onClick={check} disabled={!inp.trim()} style={{ width:"100%", background:inp.trim()?C.warm:C.border, color:inp.trim()?"#fff":C.muted, border:"none", borderRadius:12, padding:"13px", fontSize:15, fontWeight:700, cursor:inp.trim()?"pointer":"not-allowed" }}>Check Answer</button>
           :<button onClick={next} style={{ width:"100%", background:C.warm, color:"#fff", border:"none", borderRadius:12, padding:"13px", fontSize:15, fontWeight:700, cursor:"pointer" }}>{curr+1>=deck.length?"See Results":"Next →"}</button>}
    </div>
  );
}

function Match({ cards, onBack }) {
  const count=Math.min(cards.length,6);
  const [deck]=useState(()=>[...cards].sort(()=>Math.random()-.5).slice(0,count));
  const [rights]=useState(()=>[...deck].sort(()=>Math.random()-.5));
  const [ls,setLs]=useState(null);const [rs,setRs]=useState(null);const [matched,setMatched]=useState([]);const [wrong,setWrong]=useState([]);const [score,setScore]=useState(0);const [done,setDone]=useState(false);
  useEffect(()=>{
    if(ls!==null&&rs!==null){
      if(deck[ls].id===rights[rs].id){const nm=[...matched,deck[ls].id];setMatched(nm);setScore(s=>s+1);setLs(null);setRs(null);if(nm.length===deck.length)setTimeout(()=>setDone(true),500);}
      else{setWrong([ls,rs]);setTimeout(()=>{setWrong([]);setLs(null);setRs(null);},900);}
    }
  },[ls,rs]);
  if(done) return <GResults score={score} total={deck.length} onBack={onBack} />;
  const bs=(isSel,isMat,isWrong,col)=>({ background:isMat?C.greenL:isWrong?C.redL:isSel?col+"22":C.surface, border:`1.5px solid ${isMat?C.green:isWrong?C.red:isSel?col:C.border}`, borderRadius:10, padding:"10px 12px", fontSize:13, color:isMat?C.green:isWrong?C.red:C.text, cursor:isMat?"default":"pointer", textAlign:"left", lineHeight:1.4, transition:"all .2s", fontWeight:isSel?600:400, opacity:isMat?.7:1 });
  return (
    <div style={{ maxWidth:680, margin:"0 auto" }}>
      <GHeader title="Matching Pairs" score={score} curr={matched.length} total={deck.length} onBack={onBack} accent={C.green} />
      <p style={{ fontSize:13, color:C.muted, marginBottom:16, textAlign:"center" }}>Match each term to its definition</p>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:10 }}>
        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
          <p style={{ fontSize:11, fontWeight:700, color:C.muted, letterSpacing:1, textTransform:"uppercase", marginBottom:4 }}>Terms</p>
          {deck.map((c,i)=><button key={c.id} onClick={()=>{if(!matched.includes(c.id)&&!wrong.length)setLs(ls===i?null:i);}} style={bs(ls===i,matched.includes(c.id),wrong[0]===i,C.green)}>{c.question}</button>)}
        </div>
        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
          <p style={{ fontSize:11, fontWeight:700, color:C.muted, letterSpacing:1, textTransform:"uppercase", marginBottom:4 }}>Definitions</p>
          {rights.map((c,i)=><button key={c.id} onClick={()=>{if(!matched.includes(c.id)&&!wrong.length)setRs(rs===i?null:i);}} style={bs(rs===i,matched.includes(c.id),wrong[1]===i,C.green)}>{c.answer}</button>)}
        </div>
      </div>
    </div>
  );
}

function Falling({ cards, onBack }) {
  const [deck]=useState(()=>[...cards].sort(()=>Math.random()-.5));
  const [curr,setCurr]=useState(0);const [inp,setInp]=useState("");const [pos,setPos]=useState(0);const [score,setScore]=useState(0);const [lives,setLives]=useState(3);const [res,setRes]=useState(null);const [done,setDone]=useState(false);
  const inpRef=useRef();const posRef=useRef(0);
  const nextCard=useCallback((ns,nl)=>{ if(curr+1>=deck.length||nl<=0){setDone(true);return;} setTimeout(()=>{setCurr(c=>c+1);setInp("");setPos(0);posRef.current=0;setRes(null);inpRef.current?.focus();},800); },[curr,deck.length]);
  useEffect(()=>{ if(res||done)return; const id=setInterval(()=>{ posRef.current+=0.18; setPos(posRef.current); if(posRef.current>=100){clearInterval(id);setRes("missed");const nl=lives-1;setLives(nl);nextCard(score,nl);} },60); return ()=>clearInterval(id); },[curr,res,done]);
  useEffect(()=>{inpRef.current?.focus();},[curr]);
  const check=()=>{ if(res)return; const ok=inp.trim().toLowerCase()===deck[curr].answer.trim().toLowerCase(); setRes(ok?"correct":"wrong"); const ns=ok?score+1:score,nl=ok?lives:lives-1; if(ok)setScore(ns);else setLives(nl); nextCard(ns,nl); };
  if(done) return <GResults score={score} total={deck.length} onBack={onBack} msg={lives<=0?"Out of lives!":"All done!"} />;
  const card=deck[curr];
  const bc=pos<50?C.purple:pos<80?C.warm:C.red;
  const bb=pos<50?C.purpleL:pos<80?C.warmL:C.redL;
  return (
    <div style={{ maxWidth:520, margin:"0 auto" }}>
      <GHeader title="Falling Blocks" score={score} curr={curr} total={deck.length} onBack={onBack} accent={C.purple} />
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:12 }}>
        <span style={{ fontSize:14, color:C.muted }}>{"❤️".repeat(lives)}{"🖤".repeat(Math.max(0,3-lives))}</span>
        <span style={{ fontSize:13, color:C.muted }}>Type fast!</span>
      </div>
      <div style={{ background:C.surface, border:`1.5px solid ${C.border}`, borderRadius:20, height:300, position:"relative", overflow:"hidden", marginBottom:16 }}>
        <div style={{ position:"absolute", bottom:0, left:0, right:0, height:60, background:`${C.red}11`, borderTop:`2px dashed ${C.red}44` }} />
        <p style={{ position:"absolute", bottom:8, left:0, right:0, textAlign:"center", fontSize:11, color:C.red, fontWeight:700, letterSpacing:1, textTransform:"uppercase" }}>Danger Zone</p>
        <div style={{ position:"absolute", top:`${Math.min(pos,78)}%`, left:"50%", transform:"translateX(-50%)", background:bb, border:`2px solid ${bc}`, borderRadius:16, padding:"14px 20px", maxWidth:340, textAlign:"center", transition:res?"none":"top .06s linear", boxShadow:`0 4px 20px ${bc}44`, opacity:res==="correct"?0:1 }}>
          <p style={{ fontSize:12, fontWeight:700, color:bc, letterSpacing:1, textTransform:"uppercase", marginBottom:6 }}>Answer this</p>
          <p style={{ fontSize:15, color:C.text, lineHeight:1.4 }}>{card.question}</p>
        </div>
        {res && <div style={{ position:"absolute", inset:0, display:"flex", alignItems:"center", justifyContent:"center", background:res==="correct"?C.greenL+"cc":C.redL+"cc", borderRadius:20 }}>
          <p style={{ fontSize:32, fontWeight:700, color:res==="correct"?C.green:C.red }}>{res==="correct"?"✓ Correct!":res==="wrong"?`✗ ${card.answer}`:"💨 Too slow!"}</p>
        </div>}
      </div>
      <div style={{ display:"flex", gap:10 }}>
        <input ref={inpRef} value={inp} onChange={e=>setInp(e.target.value)} onKeyDown={e=>{if(e.key==="Enter")check();}} placeholder="Type answer and press Enter…"
          style={{ flex:1, border:`1.5px solid ${C.border}`, borderRadius:12, padding:"12px 16px", fontSize:15, outline:"none", color:C.text, background:C.bg }} disabled={!!res} />
        <button onClick={check} disabled={!inp.trim()||!!res} style={{ background:C.purple, color:"#fff", border:"none", borderRadius:12, padding:"12px 20px", fontSize:14, fontWeight:700, cursor:"pointer" }}>Submit</button>
      </div>
    </div>
  );
}

// ─── TOWER GAME ──────────────────────────────────────────────────────────────
function Tower({ cards, onBack }) {
  const [deck]=useState(()=>[...cards].sort(()=>Math.random()-.5));
  const [curr,setCurr]=useState(0);const [inp,setInp]=useState("");const [res,setRes]=useState(null);
  const [tower,setTower]=useState([]);const [done,setDone]=useState(false);
  const teal="#2C7A7B";
  const check=()=>{
    const ok=inp.trim().toLowerCase()===deck[curr].answer.trim().toLowerCase();
    setRes(ok?"correct":"wrong");
    if(ok)setTower(t=>[...t,{q:deck[curr].question,color:`hsl(${170+t.length*8},60%,${45-t.length*2}%)`}]);
    setTimeout(()=>{
      if(curr+1>=deck.length){setDone(true);return;}
      setCurr(c=>c+1);setInp("");setRes(null);
    },700);
  };
  if(done) return <GResults score={tower.length} total={deck.length} onBack={onBack} msg="Tower Built!" />;
  return (
    <div style={{ maxWidth:600, margin:"0 auto", display:"flex", gap:24 }}>
      <div style={{ flex:1 }}>
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16 }}>
          <button onClick={onBack} style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14 }}><Icon d={I.back} size={16} color={C.muted} /> Games</button>
          <span style={{ fontSize:14, fontWeight:700, color:teal }}>Blocks: {tower.length}</span>
        </div>
        <div style={{ background:C.surface, border:`1.5px solid ${C.border}`, borderRadius:16, padding:24, marginBottom:14 }}>
          <p style={{ fontSize:12, fontWeight:700, color:C.muted, letterSpacing:1, textTransform:"uppercase", marginBottom:10 }}>{curr+1}/{deck.length}</p>
          <p style={{ fontSize:16, color:C.text, lineHeight:1.6 }}>{deck[curr].question}</p>
        </div>
        <input value={inp} onChange={e=>{if(!res)setInp(e.target.value);}} onKeyDown={e=>{if(e.key==="Enter"&&inp.trim()&&!res)check();}}
          placeholder="Type answer…" style={{ width:"100%", border:`1.5px solid ${res==="correct"?teal:res==="wrong"?C.red:C.border}`, borderRadius:10, padding:"11px 14px", fontSize:15, outline:"none", background:res==="correct"?"#E6FFFA":res==="wrong"?C.redL:C.bg, marginBottom:10 }} />
        {res&&<p style={{ fontSize:14, fontWeight:600, color:res==="correct"?teal:C.red, marginBottom:10 }}>{res==="correct"?"🧱 Block added!":"✗ "+deck[curr].answer}</p>}
        <button onClick={()=>inp.trim()&&!res&&check()} disabled={!inp.trim()||!!res}
          style={{ width:"100%", background:inp.trim()&&!res?teal:"#ccc", color:"#fff", border:"none", borderRadius:10, padding:"12px", fontSize:15, fontWeight:700, cursor:"pointer" }}>Stack Block</button>
      </div>
      {/* Tower visual */}
      <div style={{ width:100, display:"flex", flexDirection:"column-reverse", gap:3, justifyContent:"flex-start", paddingTop:40 }}>
        {tower.map((b,i)=>(
          <div key={i} style={{ background:b.color, borderRadius:6, height:28, display:"flex", alignItems:"center", justifyContent:"center", fontSize:10, color:"#fff", fontWeight:700, overflow:"hidden", padding:"0 4px", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
            {i+1}
          </div>
        ))}
        {tower.length===0&&<p style={{ fontSize:11, color:C.muted, textAlign:"center" }}>Tower starts here</p>}
      </div>
    </div>
  );
}

// ─── SPEED RUN ────────────────────────────────────────────────────────────────
function Speedrun({ cards, onBack }) {
  const [deck]=useState(()=>[...cards].sort(()=>Math.random()-.5));
  const [curr,setCurr]=useState(0);const [inp,setInp]=useState("");const [score,setScore]=useState(0);
  const [time,setTime]=useState(60);const [done,setDone]=useState(false);const [started,setStarted]=useState(false);
  const [flash,setFlash]=useState(null);
  const gold="#D69E2E";
  useEffect(()=>{
    if(!started||done)return;
    const id=setInterval(()=>{
      setTime(t=>{if(t<=1){setDone(true);return 0;}return t-1;});
    },1000);
    return ()=>clearInterval(id);
  },[started,done]);
  const submit=()=>{
    if(!inp.trim()||done)return;
    const ok=inp.trim().toLowerCase()===deck[curr%deck.length].answer.trim().toLowerCase();
    if(ok){setScore(s=>s+1);setFlash("correct");}else{setFlash("wrong");}
    setTimeout(()=>setFlash(null),300);
    setCurr(c=>c+1);setInp("");
  };
  if(done) return <GResults score={score} total={Math.min(curr,deck.length*3)} onBack={onBack} msg={`Time's up! ${score} correct ⚡`} />;
  const card=deck[curr%deck.length];
  const pct=(time/60)*100;
  return (
    <div style={{ maxWidth:540, margin:"0 auto" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16 }}>
        <button onClick={onBack} style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14 }}><Icon d={I.back} size={16} color={C.muted} /> Games</button>
        <span style={{ fontSize:18, fontWeight:800, color:time<=10?C.red:gold }}>⏱ {time}s</span>
        <span style={{ fontSize:14, fontWeight:700, color:gold }}>✓ {score}</span>
      </div>
      <div style={{ height:8, background:C.border, borderRadius:4, marginBottom:20 }}>
        <div style={{ height:"100%", width:`${pct}%`, background:time<=10?C.red:gold, borderRadius:4, transition:"width 1s linear" }} />
      </div>
      {!started?(
        <div style={{ textAlign:"center", padding:"40px 0" }}>
          <p style={{ fontSize:48, marginBottom:16 }}>⚡</p>
          <p style={{ fontSize:18, fontWeight:700, color:C.text, marginBottom:8 }}>Speed Run</p>
          <p style={{ fontSize:14, color:C.muted, marginBottom:24 }}>Answer as many as you can in 60 seconds!</p>
          <button onClick={()=>setStarted(true)} style={{ background:gold, color:"#fff", border:"none", borderRadius:14, padding:"14px 40px", fontSize:16, fontWeight:700, cursor:"pointer" }}>Start!</button>
        </div>
      ):(
        <>
          <div style={{ background:flash==="correct"?"#E6FFFA":flash==="wrong"?C.redL:C.surface, border:`1.5px solid ${flash==="correct"?"#2C7A7B":flash==="wrong"?C.red:C.border}`, borderRadius:16, padding:24, marginBottom:14, transition:"background .2s" }}>
            <p style={{ fontSize:16, color:C.text, lineHeight:1.6 }}>{card.question}</p>
          </div>
          <div style={{ display:"flex", gap:10 }}>
            <input autoFocus value={inp} onChange={e=>setInp(e.target.value)} onKeyDown={e=>{if(e.key==="Enter")submit();}}
              placeholder="Quick! Type the answer…" style={{ flex:1, border:`1.5px solid ${C.border}`, borderRadius:10, padding:"11px 14px", fontSize:15, outline:"none" }} />
            <button onClick={submit} style={{ background:gold, color:"#fff", border:"none", borderRadius:10, padding:"11px 20px", fontSize:14, fontWeight:700, cursor:"pointer" }}>→</button>
          </div>
        </>
      )}
    </div>
  );
}

// ─── TRUE OR FALSE ────────────────────────────────────────────────────────────
function TrueFalse({ cards, onBack }) {
  const purple="#6B46C1";
  const [deck]=useState(()=>{
    const out=[];
    cards.forEach(c=>{
      out.push({statement:c.question+" — "+c.answer,correct:true,orig:c});
      const wrong=cards.filter(x=>x.id!==c.id);
      if(wrong.length>0){
        const w=wrong[Math.floor(Math.random()*wrong.length)];
        out.push({statement:c.question+" — "+w.answer,correct:false,orig:c});
      }
    });
    return out.sort(()=>Math.random()-.5).slice(0,Math.min(out.length,12));
  });
  const [curr,setCurr]=useState(0);const [score,setScore]=useState(0);const [res,setRes]=useState(null);const [done,setDone]=useState(false);
  const answer=(val)=>{
    if(res)return;
    const ok=val===deck[curr].correct;
    setRes({chosen:val,ok});
    if(ok)setScore(s=>s+1);
    setTimeout(()=>{if(curr+1>=deck.length){setDone(true);return;}setCurr(c=>c+1);setRes(null);},900);
  };
  if(done) return <GResults score={score} total={deck.length} onBack={onBack} />;
  const card=deck[curr];
  return (
    <div style={{ maxWidth:540, margin:"0 auto" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16 }}>
        <button onClick={onBack} style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14 }}><Icon d={I.back} size={16} color={C.muted} /> Games</button>
        <span style={{ fontSize:14, fontWeight:700, color:purple }}>Score: {score}</span>
      </div>
      <div style={{ height:5, background:C.border, borderRadius:3, marginBottom:20 }}>
        <div style={{ height:"100%", width:`${(curr/deck.length)*100}%`, background:purple, borderRadius:3 }} />
      </div>
      <p style={{ fontSize:12, fontWeight:700, color:C.muted, letterSpacing:1, textTransform:"uppercase", marginBottom:10 }}>{curr+1} / {deck.length}</p>
      <div style={{ background:C.surface, border:`1.5px solid ${C.border}`, borderRadius:16, padding:"28px 24px", marginBottom:24, textAlign:"center" }}>
        <p style={{ fontSize:15, color:C.text, lineHeight:1.7 }}>{card.statement}</p>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
        {[{val:true,label:"✅ True",bg:"#E6FFFA",color:"#2C7A7B"},{val:false,label:"❌ False",bg:C.redL,color:C.red}].map(opt=>{
          let bg=opt.bg; let border=`1.5px solid ${opt.color}33`;
          if(res){
            if(opt.val===card.correct){ bg="#E6FFFA"; border="2px solid #2C7A7B"; }
            else if(res.chosen===opt.val&&!res.ok){ bg=C.redL; border=`2px solid ${C.red}`; }
          }
          return <button key={String(opt.val)} onClick={()=>answer(opt.val)}
            style={{ background:bg, border, borderRadius:14, padding:"20px", fontSize:18, fontWeight:700, color:opt.color, cursor:res?"default":"pointer", transition:"all .2s" }}>{opt.label}</button>;
        })}
      </div>
      {res&&<p style={{ textAlign:"center", marginTop:14, fontSize:14, fontWeight:600, color:res.ok?"#2C7A7B":C.red }}>{res.ok?"✓ Correct!":"✗ The answer was: "+card.orig.answer}</p>}
    </div>
  );
}

// ─── MEMORY FLIP ─────────────────────────────────────────────────────────────
function Memory({ cards, onBack }) {
  const red="#C53030";
  const count=Math.min(cards.length,8);
  const [pairs]=useState(()=>{
    const selected=[...cards].sort(()=>Math.random()-.5).slice(0,count);
    const grid=[];
    selected.forEach((c,i)=>{
      grid.push({id:`q${i}`,pairId:i,text:c.question,type:"q"});
      grid.push({id:`a${i}`,pairId:i,text:c.answer,type:"a"});
    });
    return grid.sort(()=>Math.random()-.5);
  });
  const [flipped,setFlipped]=useState([]);const [matched,setMatched]=useState([]);const [moves,setMoves]=useState(0);const [done,setDone]=useState(false);
  const flip=(card)=>{
    if(matched.includes(card.id)||flipped.length===2||flipped.find(f=>f.id===card.id))return;
    const nf=[...flipped,card];
    setFlipped(nf);
    if(nf.length===2){
      setMoves(m=>m+1);
      if(nf[0].pairId===nf[1].pairId){
        const nm=[...matched,nf[0].id,nf[1].id];
        setMatched(nm);setFlipped([]);
        if(nm.length===pairs.length)setTimeout(()=>setDone(true),400);
      } else setTimeout(()=>setFlipped([]),900);
    }
  };
  if(done) return <GResults score={count} total={count} onBack={onBack} msg={`Matched all in ${moves} moves! 🃏`} />;
  return (
    <div style={{ maxWidth:640, margin:"0 auto" }}>
      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:20 }}>
        <button onClick={onBack} style={{ display:"flex", alignItems:"center", gap:6, background:"none", border:"none", cursor:"pointer", color:C.muted, fontSize:14 }}><Icon d={I.back} size={16} color={C.muted} /> Games</button>
        <span style={{ fontSize:14, color:C.muted }}>Moves: <strong style={{ color:C.text }}>{moves}</strong></span>
        <span style={{ fontSize:14, color:C.muted }}>Matched: <strong style={{ color:red }}>{matched.length/2}/{count}</strong></span>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:10 }}>
        {pairs.map(card=>{
          const isFlipped=!!flipped.find(f=>f.id===card.id);
          const isMatched=matched.includes(card.id);
          return (
            <div key={card.id} onClick={()=>flip(card)}
              style={{ height:80, borderRadius:12, cursor:isMatched?"default":"pointer", perspective:600 }}>
              <div style={{ width:"100%", height:"100%", position:"relative", transition:"transform .4s", transformStyle:"preserve-3d", transform:isFlipped||isMatched?"rotateY(180deg)":"none" }}>
                {/* Back */}
                <div style={{ position:"absolute", inset:0, backfaceVisibility:"hidden", background:isMatched?C.greenL:red, borderRadius:12, display:"flex", alignItems:"center", justifyContent:"center" }}>
                  <span style={{ fontSize:24, color:"#fff" }}>🃏</span>
                </div>
                {/* Front */}
                <div style={{ position:"absolute", inset:0, backfaceVisibility:"hidden", transform:"rotateY(180deg)", background:card.type==="q"?C.accentL:C.warmL, border:`1.5px solid ${card.type==="q"?C.accentS:C.warm}44`, borderRadius:12, display:"flex", alignItems:"center", justifyContent:"center", padding:8 }}>
                  <p style={{ fontSize:11, color:C.text, textAlign:"center", lineHeight:1.3, overflow:"hidden" }}>{card.text.slice(0,50)}{card.text.length>50?"…":""}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <style>{`.preserve-3d{transform-style:preserve-3d}`}</style>
    </div>
  );
}

// ─── MODAL ────────────────────────────────────────────────────────────────────
function Modal({ children, onClose }) {
  return (
    <div onClick={onClose} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,.4)", zIndex:1000, display:"flex", alignItems:"center", justifyContent:"center", padding:24 }}>
      <div onClick={e=>e.stopPropagation()} style={{ background:C.surface, borderRadius:20, padding:32, width:"100%", maxWidth:440, boxShadow:"0 20px 60px rgba(0,0,0,.2)" }}>
        {children}
      </div>
    </div>
  );
}
