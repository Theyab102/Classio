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
    // Strip ALL markdown symbols
    const clean = line.replace(/^#+\s*/, "").replace(/\*\*/g, "").replace(/\*/g, "").trim();
    if (!line.trim()) return <br key={i} />;
    // ALL CAPS heading line
    if (/^[A-Z][A-Z\s]{3,}$/.test(clean) || line.startsWith('# ') || line.startsWith('## ')) 
      return <p key={i} style={{ fontWeight:700, fontSize:14, marginBottom:4, marginTop:12, color:"#1a202c", letterSpacing:.3 }}>{clean}</p>;
    // Bold heading (** wrapped)
    if (/^\*\*[^*]+\*\*$/.test(line.trim())) 
      return <p key={i} style={{ fontWeight:700, fontSize:14, marginBottom:4, marginTop:10, color:"#2d3748" }}>{clean}</p>;
    // Bullet point
    if (line.startsWith('• ') || line.startsWith('- ') || line.startsWith('* ') || line.startsWith('· ')) 
      return <p key={i} style={{ paddingLeft:14, marginBottom:3, display:"flex", gap:6, lineHeight:1.6 }}><span style={{flexShrink:0, color:"#666"}}>•</span><span>{clean.replace(/^[•\-\*·]\s*/,"")}</span></p>;
    // Numbered list
    if (/^\d+\.\s/.test(line)) 
      return <p key={i} style={{ paddingLeft:14, marginBottom:3, lineHeight:1.6 }}>{clean}</p>;
    // Normal text - strip any remaining * 
    return <p key={i} style={{ marginBottom:3, lineHeight:1.6 }}>{clean}</p>;
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
  .toolbar-wrap{flex-wrap:wrap;gap:6px!important}
}
/* Hard cap on AdSense iframe so it never expands beyond our container */
ins.adsbygoogle{max-height:46px!important;overflow:hidden!important}
ins.adsbygoogle iframe{max-height:46px!important}
@keyframes bounce{0%,80%,100%{transform:scale(.8);opacity:.5}40%{transform:scale(1.1);opacity:1}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
`; 

// Global file object store — survives navigation within the session
const FILE_STORE = new Map();

// ─── INDEXEDDB FILE PERSISTENCE ──────────────────────────────────────────────
// Stores actual file blobs in the browser so they survive page refreshes
const IDB_NAME = "classio_files";
const IDB_STORE = "files";

function openIDB() {
  return new Promise((res, rej) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = e => e.target.result.createObjectStore(IDB_STORE);
    req.onsuccess = e => res(e.target.result);
    req.onerror = () => rej(req.error);
  });
}

async function idbSave(id, file) {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).put(file, id);
    await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = rej; });
  } catch(e) { console.warn("IDB save failed", e); }
}

async function idbGet(id) {
  try {
    const db = await openIDB();
    return new Promise((res, rej) => {
      const req = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(id);
      req.onsuccess = () => res(req.result || null);
      req.onerror = () => res(null);
    });
  } catch(e) { return null; }
}

async function idbDelete(id) {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).delete(id);
  } catch(e) {}
}

async function idbGetAll() {
  try {
    const db = await openIDB();
    return new Promise((res, rej) => {
      const result = {};
      const req = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).openCursor();
      req.onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) { result[cursor.key] = cursor.value; cursor.continue(); }
        else res(result);
      };
      req.onerror = () => res({});
    });
  } catch(e) { return {}; }
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────

// Defined OUTSIDE component so it is never re-created and has no closure issues
function stripBlobs(flds) {
  return (flds || []).map(fo => ({
    id: fo.id || "",
    name: fo.name || "",
    color: fo.color || "#3D5A80",
    files: (fo.files || []).map(fi => ({
      id: fi.id || "",
      name: fi.name || "",
      type: fi.type || "",
      size: fi.size || 0,
      colorIndex: fi.colorIndex || 0,
      notes: fi.notes || "",
      studyCards: fi.studyCards || [],
      uploadedAt: fi.uploadedAt || "",
      linkedFileIds: fi.linkedFileIds || [],
    })),
  }));
}

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
  const [showCharacter, setShowCharacter] = useState(false);
  const [character, setCharacter] = useState(() => {
    try { return JSON.parse(localStorage.getItem("classio_char") || "null") || { skin:"#FDDBB4", hair:"#3D2B1F", hairStyle:0, eyes:"#3D5A80", top:"#3D5A80", name:"" }; }
    catch { return { skin:"#FDDBB4", hair:"#3D2B1F", hairStyle:0, eyes:"#3D5A80", top:"#3D5A80", name:"" }; }
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // SAVE SYSTEM — localStorage primary, Firebase secondary
  // Key insight: we pass data directly into every function so there are
  // zero stale closures. No useEffect syncing, no refs for data.
  // ═══════════════════════════════════════════════════════════════════════════
  const [saveStatus, setSaveStatus] = useState("idle");
  const saveTimer    = useRef(null);
  const statusTimer  = useRef(null);

  // Write clean JSON to localStorage + Firebase
  // flds and u are passed directly — never read from closure/ref
  function persist(flds, u) {
    const clean = stripBlobs(flds);

    // 1. localStorage — instant, works offline, no permissions needed
    try { localStorage.setItem("classio_v2", JSON.stringify(clean)); } catch(e) { console.error("LS save error:", e); }

    // 2. Firebase — best effort background sync
    if (u?.uid) {
      // Correct Firestore path: collection="users", document=uid
      setDoc(doc(db, "users", u.uid), { folders: clean }, { merge: true })
        .then(() => {
          clearTimeout(statusTimer.current);
          setSaveStatus("saved");
          statusTimer.current = setTimeout(() => setSaveStatus("idle"), 2000);
        })
        .catch(e => {
          // localStorage already saved — Firebase failing is not critical
          console.warn("Firebase sync failed (data saved locally):", e.code, e.message);
          clearTimeout(statusTimer.current);
          setSaveStatus("saved"); // local save succeeded
          statusTimer.current = setTimeout(() => setSaveStatus("idle"), 2000);
        });
    } else {
      clearTimeout(statusTimer.current);
      setSaveStatus("saved");
      statusTimer.current = setTimeout(() => setSaveStatus("idle"), 2000);
    }
  }

  // Debounce: wait 500ms after last change, then persist
  // Takes flds AND u directly so nothing is stale
  function scheduleSave(flds, u) {
    setSaveStatus("saving");
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => persist(flds, u), 500);
  }

  // Attach file blobs (non-serialisable) back onto plain folder data
  async function attachBlobs(raw) {
    const stored = await idbGetAll();
    return (raw || []).map(fo => ({
      ...fo,
      files: (fo.files || []).map(fi => {
        const blob = stored[fi.id] || FILE_STORE.get(fi.id) || null;
        if (blob) FILE_STORE.set(fi.id, blob);
        return { ...fi, _fileObj: blob };
      }),
    }));
  }

  // Startup: load localStorage instantly, then Firebase if logged in
  useEffect(() => {
    const local = localStorage.getItem("classio_v2");
    if (local) {
      try { attachBlobs(JSON.parse(local)).then(r => setFolders(p => p.length === 0 ? r : p)); } catch {}
    }
    const unsub = onAuthStateChanged(auth, async (u) => {
      setUser(u);
      if (u) {
        setIsGuest(false);
        try {
          const snap = await getDoc(doc(db, "users", u.uid));
          const raw  = snap?.exists() ? (snap.data().folders || []) : [];
          if (raw.length > 0) {
            const r = await attachBlobs(raw);
            setFolders(r);
            try { localStorage.setItem("classio_v2", JSON.stringify(stripBlobs(r))); } catch {}
          }
        } catch(e) { console.warn("Firebase load failed, using local data:", e.message); }
      }
    });
    return unsub;
  }, []);

  // THE ONE entry point for ALL data changes — always call this instead of setFolders
  function applyAndSave(flds) {
    setFolders(flds);
    scheduleSave(flds, user); // user from component scope — always current at call time
  }

  const updateFolder = (updated) => {
    setFolders(prev => {
      const next = prev.map(f => f.id === updated.id ? updated : f);
      scheduleSave(next, user);
      return next;
    });
    if (activeFolder?.id === updated.id) setActiveFolder(updated);
  };

  const updateFile = (folderId, updated) => {
    const withObj = { ...updated, _fileObj: updated._fileObj || FILE_STORE.get(updated.id) || null };
    setFolders(prev => {
      const next = prev.map(f => f.id === folderId
        ? { ...f, files: f.files.map(fi => fi.id === withObj.id ? withObj : fi) }
        : f);
      scheduleSave(next, user);
      return next;
    });
    setActiveFile(withObj);
    setActiveFolder(prev => prev ? { ...prev, files: prev.files.map(fi => fi.id === withObj.id ? withObj : fi) } : prev);
  };

  const deleteFolder = (folderId) => {
    setFolders(prev => {
      const folder = prev.find(f => f.id === folderId);
      if (folder) (folder.files || []).forEach(f => { idbDelete(f.id); FILE_STORE.delete(f.id); });
      const next = prev.filter(f => f.id !== folderId);
      scheduleSave(next, user);
      return next;
    });
  };

  // Keep setFoldersSave for places that still use it (folder creation etc.)
  const setFoldersSave = (flds) => applyAndSave(flds);

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
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "'DM Sans', sans-serif", paddingBottom: 50 }}>
      <style>{GS}</style>
      <Header user={isGuest ? { displayName: guestName, photoURL: null } : user} saveStatus={saveStatus} isGuest={isGuest} onSignOut={isGuest ? handleGuestSignOut : () => signOut(auth)} character={character} onOpenCharacter={() => setShowCharacter(true)} />
      {showCharacter && <CharacterModal character={character} onChange={c => { setCharacter(c); localStorage.setItem("classio_char", JSON.stringify(c)); }} onClose={() => setShowCharacter(false)} />}
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
              style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:16, padding:20, cursor:"pointer", boxShadow:"0 2px 8px rgba(0,0,0,.05)", position:"relative" }}>
              {/* Delete folder button */}
              <button
                onClick={e => {
                  e.stopPropagation();
                  if (window.confirm(`Delete "${folder.name}" and all its files?`)) deleteFolder(folder.id);
                }}
                style={{ position:"absolute", top:10, right:10, width:26, height:26, borderRadius:"50%", background:"transparent", border:"none", cursor:"pointer", display:"flex", alignItems:"center", justifyContent:"center", opacity:0.4, fontSize:16, color:C.muted, lineHeight:1 }}
                onMouseEnter={e => e.currentTarget.style.opacity=1}
                onMouseLeave={e => e.currentTarget.style.opacity=0.4}
                title="Delete folder">
                ×
              </button>
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
    <div style={{
      position:"fixed", bottom:0, left:0, right:0, zIndex:999,
      height:50, maxHeight:50, overflow:"hidden",
      background:C.surface, borderTop:`1px solid ${C.border}`,
      display:"flex", alignItems:"center", justifyContent:"center",
    }}>
      <div style={{ position:"relative", width:"100%", maxWidth:728, height:46, overflow:"hidden" }}>
        <ins className="adsbygoogle"
          style={{ display:"block", width:"100%", height:46, overflow:"hidden" }}
          data-ad-client="ca-pub-5802600279565250"
          data-ad-slot="7527000448"
          data-ad-format="horizontal"
          data-full-width-responsive="false" />
      </div>
    </div>
  );
}

// ─── HEADER ───────────────────────────────────────────────────────────────────
function Header({ user, saveStatus, isGuest, onSignOut, character, onOpenCharacter }) {
  return (
    <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"0 16px", height:56, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        <div style={{ width:30, height:30, background:C.accent, borderRadius:9, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <Icon d={I.sparkle} size={15} color="#fff" sw={2} />
        </div>
        <span style={{ fontFamily:"'Fraunces',serif", fontSize:20, fontWeight:700, color:C.text, letterSpacing:-0.5 }}>Classio</span>
      </div>
      <div style={{ display:"flex", alignItems:"center", gap:8 }}>
        {!isGuest && saveStatus !== "idle" && (
          <span style={{ fontSize:12, fontWeight:600,
            color: saveStatus==="saved" ? C.green : saveStatus==="error" ? "#e53e3e" : C.muted }}>
            {saveStatus==="saving" ? "Saving…" : saveStatus==="saved" ? "✓ Saved" : "⚠ Save failed"}
          </span>
        )}
        {isGuest && <span className="desktop-only" style={{ fontSize:11, background:C.warmL, color:C.warm, border:`1px solid ${C.warm}44`, borderRadius:20, padding:"3px 8px", fontWeight:600 }}>Guest</span>}
        <button onClick={onOpenCharacter} title="Edit my avatar"
          style={{ background:"none", border:"none", borderRadius:"50%", width:36, height:36, padding:0, cursor:"pointer", overflow:"hidden", flexShrink:0, boxShadow:"0 2px 8px rgba(0,0,0,.15)" }}>
          <MiniAvatar character={character} size={36} />
        </button>
        <span className="desktop-only" style={{ fontSize:13, fontWeight:600, color:C.text }}>{isGuest ? user?.displayName : user?.displayName?.split(" ")[0]}</span>
        <button onClick={onSignOut} className="hov"
          style={{ fontSize:12, color:C.muted, background:"none", border:`1px solid ${C.border}`, borderRadius:7, padding:"4px 9px", cursor:"pointer", whiteSpace:"nowrap" }}>{isGuest ? "Exit" : "Sign out"}</button>
      </div>
    </div>
  );
}

// ─── SPLASH / SIGN IN ─────────────────────────────────────────────────────────
// ─── AVATAR SVG ──────────────────────────────────────────────────────────────
// Human-shaped bust avatar with 3D shading, Snapchat-inspired style
function MiniAvatar({ character: ch, size = 40 }) {
  const s = size;
  const cx = s * 0.5;
  const skinDark  = shadeSkin(ch.skin, -25);
  const skinLight = shadeSkin(ch.skin, 30);
  const hairDark  = shadeHex(ch.hair, -30);
  const topDark   = shadeHex(ch.top,  -20);

  // 10 hair styles rendered as SVG
  const HAIR = {
    0: <>  {/* Buzz / short crop */}
      <ellipse cx={cx} cy={s*.19} rx={s*.21} ry={s*.135} fill={ch.hair}/>
      <rect x={s*.29} y={s*.19} width={s*.42} height={s*.09} fill={ch.hair}/>
    </>,
    1: <>  {/* Side-part clean */}
      <ellipse cx={cx} cy={s*.17} rx={s*.23} ry={s*.15} fill={hairDark}/>
      <ellipse cx={cx} cy={s*.17} rx={s*.21} ry={s*.12} fill={ch.hair}/>
      <rect x={s*.29} y={s*.19} width={s*.42} height={s*.1} fill={ch.hair}/>
      <path d={`M${s*.27} ${s*.17} Q${s*.38} ${s*.1} ${s*.6} ${s*.15}`} stroke={hairDark} strokeWidth={s*.018} fill="none"/>
    </>,
    2: <>  {/* Waves / medium */}
      <ellipse cx={cx} cy={s*.17} rx={s*.23} ry={s*.155} fill={ch.hair}/>
      <rect x={s*.27} y={s*.2} width={s*.07} height={s*.26} rx={s*.035} fill={ch.hair}/>
      <rect x={s*.66} y={s*.2} width={s*.07} height={s*.26} rx={s*.035} fill={ch.hair}/>
      <path d={`M${s*.27} ${s*.28} Q${s*.3} ${s*.35} ${s*.27} ${s*.42}`} stroke={hairDark} strokeWidth={s*.015} fill="none"/>
      <path d={`M${s*.73} ${s*.28} Q${s*.7} ${s*.35} ${s*.73} ${s*.42}`} stroke={hairDark} strokeWidth={s*.015} fill="none"/>
    </>,
    3: <>  {/* Long straight */}
      <ellipse cx={cx} cy={s*.16} rx={s*.24} ry={s*.155} fill={ch.hair}/>
      <rect x={s*.26} y={s*.19} width={s*.07} height={s*.46} rx={s*.03} fill={ch.hair}/>
      <rect x={s*.67} y={s*.19} width={s*.07} height={s*.46} rx={s*.03} fill={ch.hair}/>
      <rect x={s*.27} y={s*.19} width={s*.46} height={s*.1} fill={ch.hair}/>
      <ellipse cx={s*.295} cy={s*.65} rx={s*.04} ry={s*.06} fill={hairDark}/>
      <ellipse cx={s*.705} cy={s*.65} rx={s*.04} ry={s*.06} fill={hairDark}/>
    </>,
    4: <>  {/* High bun */}
      <ellipse cx={cx} cy={s*.2} rx={s*.22} ry={s*.13} fill={ch.hair}/>
      <rect x={s*.29} y={s*.2} width={s*.42} height={s*.09} fill={ch.hair}/>
      <circle cx={cx} cy={s*.085} r={s*.1} fill={ch.hair}/>
      <circle cx={cx} cy={s*.085} r={s*.065} fill={hairDark} opacity={.4}/>
      <circle cx={cx} cy={s*.085} r={s*.03} fill={skinLight} opacity={.3}/>
    </>,
    5: <>  {/* Ponytail */}
      <ellipse cx={cx} cy={s*.17} rx={s*.23} ry={s*.14} fill={ch.hair}/>
      <rect x={s*.29} y={s*.19} width={s*.42} height={s*.09} fill={ch.hair}/>
      <ellipse cx={s*.79} cy={s*.25} rx={s*.055} ry={s*.15} fill={ch.hair} transform={`rotate(-15,${s*.79},${s*.25})`}/>
      <ellipse cx={s*.79} cy={s*.25} rx={s*.025} ry={s*.1} fill={hairDark} opacity={.4} transform={`rotate(-15,${s*.79},${s*.25})`}/>
    </>,
    6: <>  {/* Afro */}
      {[...Array(14)].map((_,i) => {
        const a = (i/14)*Math.PI*2;
        return <circle key={i} cx={cx+Math.cos(a)*s*.19} cy={s*.19+Math.sin(a)*s*.15} r={s*.07} fill={ch.hair}/>;
      })}
      <ellipse cx={cx} cy={s*.19} rx={s*.2} ry={s*.16} fill={ch.hair}/>
    </>,
    7: <>  {/* Braids */}
      <ellipse cx={cx} cy={s*.17} rx={s*.23} ry={s*.15} fill={ch.hair}/>
      {[0,1,2].map(i=><>
        <rect key={`l${i}`} x={s*(.3+i*.04)} y={s*.22} width={s*.025} height={s*.38} rx={s*.012} fill={i%2===0?ch.hair:hairDark}/>
        <rect key={`r${i}`} x={s*(.645+i*.04)} y={s*.22} width={s*.025} height={s*.38} rx={s*.012} fill={i%2===0?hairDark:ch.hair}/>
      </>)}
    </>,
    8: <>  {/* Curly bob */}
      {[...Array(9)].map((_,i) => <circle key={i} cx={s*(.28+i*.055)} cy={s*.18} r={s*.055} fill={ch.hair}/>)}
      <ellipse cx={cx} cy={s*.21} rx={s*.22} ry={s*.1} fill={ch.hair}/>
      <rect x={s*.27} y={s*.21} width={s*.07} height={s*.16} rx={s*.035} fill={ch.hair}/>
      <rect x={s*.66} y={s*.21} width={s*.07} height={s*.16} rx={s*.035} fill={ch.hair}/>
    </>,
    9: <>  {/* Locs / dreadlocks */}
      <ellipse cx={cx} cy={s*.17} rx={s*.23} ry={s*.15} fill={ch.hair}/>
      {[0,1,2,3,4].map(i=><rect key={`l${i}`} x={s*(.27+i*.046)} y={s*.21} width={s*.03} height={s*(.25+i*.04)} rx={s*.015} fill={i%2===0?ch.hair:hairDark}/>)}
      {[0,1,2].map(i=><rect key={`r${i}`} x={s*(.65+i*.046)} y={s*.21} width={s*.03} height={s*(.28+i*.04)} rx={s*.015} fill={i%2===0?hairDark:ch.hair}/>)}
    </>,
  };

  const MOUTH = {
    0: <path d={`M${s*.4} ${s*.515} Q${cx} ${s*.565} ${s*.6} ${s*.515}`} stroke="#c07060" strokeWidth={s*.025} fill="none" strokeLinecap="round"/>,
    1: <path d={`M${s*.4} ${s*.535} Q${cx} ${s*.49} ${s*.6} ${s*.535}`} stroke="#c07060" strokeWidth={s*.025} fill="none" strokeLinecap="round"/>,
    2: <ellipse cx={cx} cy={s*.528} rx={s*.08} ry={s*.032} fill="#c07060"/>,
    3: <>
      <path d={`M${s*.38} ${s*.51} Q${cx} ${s*.585} ${s*.62} ${s*.51}`} fill="#b05040"/>
      <path d={`M${s*.38} ${s*.51} Q${cx} ${s*.585} ${s*.62} ${s*.51}`} fill="white" opacity={.5}/>
      <path d={`M${s*.38} ${s*.51} Q${cx} ${s*.585} ${s*.62} ${s*.51}`} stroke="#c07060" strokeWidth={s*.022} fill="none" strokeLinecap="round"/>
    </>,
    4: <>
      <path d={`M${s*.4} ${s*.515} Q${cx} ${s*.555} ${s*.6} ${s*.515}`} stroke="#c07060" strokeWidth={s*.022} fill="none" strokeLinecap="round"/>
      <rect x={s*.44} y={s*.515} width={s*.02} height={s*.03} rx={s*.01} fill="#c07060" opacity={.5}/>
      <rect x={s*.49} y={s*.515} width={s*.02} height={s*.03} rx={s*.01} fill="#c07060" opacity={.5}/>
    </>,
  };

  const BROW = {
    0: <><path d={`M${s*.33} ${s*.305} Q${s*.4} ${s*.275} ${s*.46} ${s*.29}`} stroke={hairDark} strokeWidth={s*.023} fill="none" strokeLinecap="round"/>
        <path d={`M${s*.54} ${s*.29} Q${s*.6} ${s*.275} ${s*.67} ${s*.305}`} stroke={hairDark} strokeWidth={s*.023} fill="none" strokeLinecap="round"/></>,
    1: <><line x1={s*.33} y1={s*.295} x2={s*.46} y2={s*.3} stroke={hairDark} strokeWidth={s*.022} strokeLinecap="round"/>
        <line x1={s*.54} y1={s*.3} x2={s*.67} y2={s*.295} stroke={hairDark} strokeWidth={s*.022} strokeLinecap="round"/></>,
    2: <><path d={`M${s*.33} ${s*.3} Q${s*.4} ${s*.265} ${s*.46} ${s*.285}`} stroke={hairDark} strokeWidth={s*.03} fill="none" strokeLinecap="round"/>
        <path d={`M${s*.54} ${s*.285} Q${s*.6} ${s*.265} ${s*.67} ${s*.3}`} stroke={hairDark} strokeWidth={s*.03} fill="none" strokeLinecap="round"/></>,
    3: <><path d={`M${s*.33} ${s*.29} Q${s*.4} ${s*.31} ${s*.46} ${s*.3}`} stroke={hairDark} strokeWidth={s*.022} fill="none" strokeLinecap="round"/>
        <path d={`M${s*.54} ${s*.3} Q${s*.6} ${s*.31} ${s*.67} ${s*.29}`} stroke={hairDark} strokeWidth={s*.022} fill="none" strokeLinecap="round"/></>,
  };

  const eye = (ex, ey) => {
    const es = ch.eyeShape || 0;
    const whites = es===2
      ? <path d={`M${ex-s*.058} ${ey} Q${ex} ${ey-s*.075} ${ex+s*.058} ${ey} Q${ex} ${ey+s*.04} ${ex-s*.058} ${ey}`} fill="white"/>
      : <ellipse cx={ex} cy={ey} rx={s*.058} ry={es===1?s*.04:s*.055} fill="white"/>;
    return <>
      {whites}
      <ellipse cx={ex} cy={ey} rx={s*.032} ry={s*.032} fill={ch.eyes}/>
      <ellipse cx={ex} cy={ey} rx={s*.018} ry={s*.018} fill="#111" opacity={.85}/>
      <circle cx={ex+s*.018} cy={ey-s*.016} r={s*.009} fill="white"/>
      <circle cx={ex-s*.01} cy={ey+s*.01} r={s*.005} fill="white" opacity={.6}/>
    </>;
  };

  const nose = <><path d={`M${cx} ${s*.415} Q${s*.545} ${s*.455} ${s*.535} ${s*.49}`} stroke={skinDark} strokeWidth={s*.016} fill="none" opacity={.5}/>
    <ellipse cx={s*.465} cy={s*.495} rx={s*.018} ry={s*.01} fill={skinDark} opacity={.3}/>
    <ellipse cx={s*.535} cy={s*.495} rx={s*.018} ry={s*.01} fill={skinDark} opacity={.3}/></>;

  const acc = ch.accessory===1
    ? <><rect x={s*.3} y={s*.33} width={s*.16} height={s*.1} rx={s*.04} fill="none" stroke="#444" strokeWidth={s*.02}/>
        <rect x={s*.54} y={s*.33} width={s*.16} height={s*.1} rx={s*.04} fill="none" stroke="#444" strokeWidth={s*.02}/>
        <line x1={s*.46} y1={s*.38} x2={s*.54} y2={s*.38} stroke="#444" strokeWidth={s*.016}/></>
    : ch.accessory===2
    ? <ellipse cx={cx} cy={s*.185} rx={s*.27} ry={s*.075} fill={ch.hair} opacity={.95}/>
    : ch.accessory===3
    ? <><path d={`M${s*.28} ${s*.32} Q${cx} ${s*.26} ${s*.72} ${s*.32}`} fill={ch.top} opacity={.85}/>
        <rect x={s*.35} y={s*.3} width={s*.3} height={s*.06} rx={s*.02} fill={ch.top}/></>
    : null;

  const bodyGrad = `url(#bg${ch.top.replace('#','')})`;

  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} style={{display:"block"}}>
      <defs>
        <radialGradient id={`face${s}`} cx="40%" cy="35%" r="65%">
          <stop offset="0%" stopColor={skinLight}/>
          <stop offset="60%" stopColor={ch.skin}/>
          <stop offset="100%" stopColor={skinDark}/>
        </radialGradient>
        <radialGradient id={`body${s}`} cx="40%" cy="30%" r="70%">
          <stop offset="0%" stopColor={shadeHex(ch.top,25)}/>
          <stop offset="55%" stopColor={ch.top}/>
          <stop offset="100%" stopColor={topDark}/>
        </radialGradient>
        <radialGradient id={`bg${s}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor={shadeHex(ch.bg||"#e8f4ff",15)}/>
          <stop offset="100%" stopColor={ch.bg||"#e8f4ff"}/>
        </radialGradient>
        <clipPath id={`clip${s}`}><circle cx={cx} cy={cx} r={cx}/></clipPath>
      </defs>

      {/* Background */}
      <circle cx={cx} cy={cx} r={cx} fill={`url(#bg${s})`}/>

      {/* Shoulders / body — human bust shape */}
      <path d={`M${s*.1} ${s*1.02} Q${s*.18} ${s*.72} ${s*.32} ${s*.67} L${s*.38} ${s*.64} Q${cx} ${s*.62} ${s*.62} ${s*.64} L${s*.68} ${s*.67} Q${s*.82} ${s*.72} ${s*.9} ${s*1.02} Z`}
        fill={`url(#body${s})`} clipPath={`url(#clip${s})`}/>
      {/* Shoulder highlight */}
      <path d={`M${s*.18} ${s*.88} Q${s*.28} ${s*.71} ${s*.38} ${s*.66}`} stroke={shadeHex(ch.top,30)} strokeWidth={s*.025} fill="none" opacity={.4} clipPath={`url(#clip${s})`}/>

      {/* Neck */}
      <path d={`M${s*.42} ${s*.56} L${s*.42} ${s*.64} Q${cx} ${s*.67} ${s*.58} ${s*.64} L${s*.58} ${s*.56}`}
        fill={`url(#face${s})`} clipPath={`url(#clip${s})`}/>
      {/* Neck shadow */}
      <ellipse cx={cx} cy={s*.64} rx={s*.1} ry={s*.025} fill={skinDark} opacity={.25} clipPath={`url(#clip${s})`}/>

      {/* Ear left */}
      <ellipse cx={s*.28} cy={s*.39} rx={s*.04} ry={s*.055} fill={`url(#face${s})`}/>
      <ellipse cx={s*.285} cy={s*.39} rx={s*.02} ry={s*.032} fill={skinDark} opacity={.3}/>
      {/* Ear right */}
      <ellipse cx={s*.72} cy={s*.39} rx={s*.04} ry={s*.055} fill={`url(#face${s})`}/>
      <ellipse cx={s*.715} cy={s*.39} rx={s*.02} ry={s*.032} fill={skinDark} opacity={.3}/>

      {/* Hair back layer */}
      {HAIR[ch.hairStyle||0]}

      {/* Face */}
      <ellipse cx={cx} cy={s*.38} rx={s*.205} ry={s*.225} fill={`url(#face${s})`}/>
      {/* Face edge shadow */}
      <ellipse cx={cx} cy={s*.38} rx={s*.205} ry={s*.225} fill="none" stroke={skinDark} strokeWidth={s*.012} opacity={.18}/>

      {/* Blush */}
      {ch.blush && <>
        <ellipse cx={s*.33} cy={s*.46} rx={s*.055} ry={s*.03} fill="#ff6b9d" opacity={.25}/>
        <ellipse cx={s*.67} cy={s*.46} rx={s*.055} ry={s*.03} fill="#ff6b9d" opacity={.25}/>
      </>}

      {/* Eyebrows */}
      {BROW[ch.eyebrow||0]}
      {/* Eyes */}
      {eye(s*.385, s*.365)}
      {eye(s*.615, s*.365)}
      {/* Nose */}
      {nose}
      {/* Mouth */}
      {MOUTH[ch.mouth||0]}

      {/* Lip gloss */}
      {ch.lips && <ellipse cx={cx} cy={s*.515} rx={s*.055} ry={s*.016} fill={ch.lipColor||"#e07070"} opacity={.55}/>}

      {/* Accessory */}
      {acc}
    </svg>
  );
}

function shadeSkin(hex, pct) { return shadeHex(hex, pct); }
function shadeHex(hex, pct) {
  const n = parseInt(hex.replace('#',''),16);
  const r = Math.min(255,Math.max(0,((n>>16)&255)+pct));
  const g = Math.min(255,Math.max(0,((n>>8)&255)+pct));
  const b = Math.min(255,Math.max(0,(n&255)+pct));
  return `rgb(${r},${g},${b})`;
}

// ─── CHARACTER MODAL ─────────────────────────────────────────────────────────
function CharacterModal({ character, onChange, onClose }) {
  const ch = character;
  const [tab, setTab] = useState("face");

  const SKINS  = ["#FDDBB4","#F5C89A","#FFCBA4","#E8A87C","#D4956A","#C68642","#A0693A","#8D5524","#6B3A1F","#F4D6C8"];
  const HAIRS  = ["#0d0d0d","#2C1810","#4A2912","#7B4F2C","#B5651D","#C9A96E","#E8D5B0","#F2E6C8","#C0392B","#E74C3C","#8E44AD","#2980B9","#27AE60","#F39C12","#1ABC9C","#fd79a8","#636e72","#95a5a6"];
  const EYES   = ["#1a3a5c","#2980B9","#74b9ff","#27AE60","#52BE80","#8B6914","#C8A84B","#2C2C2C","#555555","#C0392B","#8E44AD","#00b894","#e17055"];
  const TOPS   = ["#2C3E50","#34495E","#E74C3C","#C0392B","#E67E22","#F39C12","#27AE60","#16A085","#2980B9","#8E44AD","#fd79a8","#FFFFFF","#ECF0F1","#BDC3C7"];
  const LIP_COLORS = ["#C0392B","#E74C3C","#e07070","#fd79a8","#8E44AD","#D35400","#b5451b","#922B21"];
  const BG = ["#e8f4ff","#dfe6e9","#ffeaa7","#d5f5e3","#f8d7e3","#e8daef","#ffddd2","#c8e6c9","#fff3e0","#e3f2fd","#1a1a2e","#2d3436"];
  const HAIR_STYLES = ["Buzz","Side-part","Waves","Long","Bun","Ponytail","Afro","Braids","Curly bob","Locs"];
  const EYE_SHAPES  = ["Round","Almond","Cat-eye"];
  const BROWS       = ["Arched","Straight","Thick","Worried"];
  const MOUTHS      = ["Smile","Frown","Neutral","Big smile","Smirk"];
  const ACCESSORIES = ["None","Glasses","Hat","Cap"];
  const TOP_STYLES  = ["T-Shirt","Hoodie","Jacket","Dress","Suit","Crop top"];

  const TABS = [
    { id:"face",  icon:"😊", label:"Face"   },
    { id:"hair",  icon:"💇", label:"Hair"   },
    { id:"style", icon:"👕", label:"Style"  },
    { id:"extra", icon:"✨", label:"Extras" },
  ];

  const Swatches = ({ values, field, size=26 }) => (
    <div style={{ display:"flex", gap:5, flexWrap:"wrap" }}>
      {values.map(v => (
        <button key={v} onClick={() => onChange({...ch,[field]:v})} style={{
          width:size, height:size, borderRadius:"50%", background:v,
          border:`3px solid ${ch[field]===v?"#fff":"transparent"}`,
          outline:`2px solid ${ch[field]===v?C.accent:"transparent"}`,
          cursor:"pointer", transition:"transform .15s",
          transform:ch[field]===v?"scale(1.2)":"scale(1)",
          boxShadow:"0 1px 4px rgba(0,0,0,.2)"
        }}/>
      ))}
    </div>
  );

  const Pills = ({ values, field }) => (
    <div style={{ display:"flex", gap:5, flexWrap:"wrap" }}>
      {values.map((v,i) => (
        <button key={i} onClick={() => onChange({...ch,[field]:i})} style={{
          padding:"5px 12px", fontSize:11, fontWeight:700, borderRadius:20,
          border:"none", cursor:"pointer", transition:"all .15s",
          background:ch[field]===i?"#FFFC00":C.border,
          color:ch[field]===i?"#000":C.muted,
          boxShadow:ch[field]===i?"0 2px 8px rgba(0,0,0,.2)":"none",
          transform:ch[field]===i?"scale(1.05)":"scale(1)"
        }}>{v}</button>
      ))}
    </div>
  );

  const Toggle = ({ label, field, emoji }) => (
    <button onClick={() => onChange({...ch,[field]:!ch[field]})} style={{
      display:"flex", alignItems:"center", gap:8, padding:"8px 14px",
      borderRadius:20, border:"none", cursor:"pointer",
      background:ch[field]?"#FFFC00":C.border,
      color:ch[field]?"#000":C.muted, fontWeight:700, fontSize:12,
      boxShadow:ch[field]?"0 2px 8px rgba(0,0,0,.2)":"none"
    }}>{emoji} {label}: {ch[field]?"ON":"OFF"}</button>
  );

  return (
    <div onClick={onClose} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,.6)", zIndex:2000, display:"flex", alignItems:"center", justifyContent:"center", padding:12 }}>
      <div onClick={e=>e.stopPropagation()} style={{
        background:"#f7f7f7", borderRadius:28, width:"100%", maxWidth:500,
        maxHeight:"93vh", overflow:"hidden", display:"flex", flexDirection:"column",
        boxShadow:"0 32px 100px rgba(0,0,0,.4)"
      }}>

        {/* Top bar — Snapchat yellow */}
        <div style={{ background:"#FFFC00", padding:"14px 20px 10px", display:"flex", alignItems:"center", justifyContent:"space-between" }}>
          <span style={{ fontFamily:"'Fraunces',serif", fontSize:20, fontWeight:900, color:"#000" }}>My Avatar</span>
          <button onClick={onClose} style={{ background:"rgba(0,0,0,.12)", border:"none", borderRadius:"50%", width:32, height:32, fontSize:18, cursor:"pointer", color:"#000", fontWeight:700, display:"flex", alignItems:"center", justifyContent:"center" }}>×</button>
        </div>

        {/* Preview strip */}
        <div style={{ background:"linear-gradient(135deg,#667eea22,#764ba222)", padding:"16px 0 10px", display:"flex", flexDirection:"column", alignItems:"center", gap:8 }}>
          <div style={{ borderRadius:999, overflow:"hidden", boxShadow:"0 8px 32px rgba(0,0,0,.2)", border:"3px solid #FFFC00" }}>
            <MiniAvatar character={ch} size={120}/>
          </div>
          <input value={ch.name||""} onChange={e=>onChange({...ch,name:e.target.value})} placeholder="Your name…"
            style={{ border:"2px solid #FFFC00", borderRadius:20, padding:"5px 16px", fontSize:13, fontWeight:700, outline:"none", color:"#000", background:"white", textAlign:"center", width:160 }}/>
        </div>

        {/* Tab bar */}
        <div style={{ display:"flex", background:"#fff", borderBottom:"2px solid #eee" }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              flex:1, padding:"10px 4px", border:"none", cursor:"pointer",
              background:tab===t.id?"#FFFC00":"#fff",
              fontWeight:700, fontSize:11, color:tab===t.id?"#000":"#888",
              borderBottom:tab===t.id?"3px solid #000":"3px solid transparent",
              transition:"all .15s"
            }}>{t.icon}<br/>{t.label}</button>
          ))}
        </div>

        {/* Options */}
        <div style={{ flex:1, overflowY:"auto", padding:"16px 18px", display:"flex", flexDirection:"column", gap:16 }}>

          {tab==="face" && <>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>SKIN TONE</p>
              <Swatches values={SKINS} field="skin"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>EYE SHAPE</p>
              <Pills values={EYE_SHAPES} field="eyeShape"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>EYE COLOUR</p>
              <Swatches values={EYES} field="eyes"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>EYEBROWS</p>
              <Pills values={BROWS} field="eyebrow"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>EXPRESSION</p>
              <Pills values={MOUTHS} field="mouth"/>
            </div>
          </>}

          {tab==="hair" && <>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>HAIR STYLE</p>
              <Pills values={HAIR_STYLES} field="hairStyle"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>HAIR COLOUR</p>
              <Swatches values={HAIRS} field="hair"/>
            </div>
          </>}

          {tab==="style" && <>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>OUTFIT</p>
              <Pills values={TOP_STYLES} field="topStyle"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>OUTFIT COLOUR</p>
              <Swatches values={TOPS} field="top"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>BACKGROUND</p>
              <Swatches values={BG} field="bg"/>
            </div>
            <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>ACCESSORIES</p>
              <Pills values={ACCESSORIES} field="accessory"/>
            </div>
          </>}

          {tab==="extra" && <>
            <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
              <Toggle label="Blush" field="blush" emoji="🌸"/>
              <Toggle label="Lip gloss" field="lips" emoji="💋"/>
            </div>
            {ch.lips && <div>
              <p style={{ fontSize:11, fontWeight:800, color:"#555", letterSpacing:1, marginBottom:8 }}>LIP COLOUR</p>
              <Swatches values={LIP_COLORS} field="lipColor"/>
            </div>}
            <button onClick={()=>onChange({skin:"#FDDBB4",hair:"#3D2B1F",hairStyle:0,eyes:"#2980B9",top:"#2C3E50",bg:"#e8f4ff",mouth:0,eyebrow:0,eyeShape:0,accessory:0,topStyle:0,blush:false,lips:false,lipColor:"#e07070",name:ch.name})}
              style={{ padding:"8px 20px", background:"#eee", border:"none", borderRadius:20, fontSize:12, fontWeight:700, cursor:"pointer", color:"#555", alignSelf:"flex-start" }}>
              🔄 Reset to default
            </button>
          </>}
        </div>

        {/* Done button */}
        <div style={{ padding:"10px 18px 14px", background:"#fff", borderTop:"1px solid #eee" }}>
          <button onClick={onClose} style={{ width:"100%", background:"#FFFC00", color:"#000", border:"none", borderRadius:14, padding:"12px", fontSize:15, fontWeight:900, cursor:"pointer", boxShadow:"0 4px 16px rgba(0,0,0,.15)" }}>
            ✓ Done
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── LINK FILES BUTTON ────────────────────────────────────────────────────────
function LinkBtn({ file, allFiles, onSave }) {
  const [open, setOpen] = useState(false);
  const others = allFiles.filter(f => f.id !== file.id);
  const linked = file.linkedFileIds || [];

  if (others.length === 0) return null;

  return (
    <>
      <button onClick={() => setOpen(true)} className="hov"
        style={{ display:"flex", alignItems:"center", gap:5, background:linked.length>0?C.accentL:"none", color:C.accent, border:`1px solid ${C.border}`, borderRadius:8, padding:"5px 10px", fontSize:12, fontWeight:600, cursor:"pointer", whiteSpace:"nowrap" }}>
        🔗 {linked.length > 0 ? `${linked.length} linked` : "Link files"}
      </button>
      {open && (
        <div onClick={() => setOpen(false)} style={{ position:"fixed", inset:0, background:"rgba(0,0,0,.45)", zIndex:2000, display:"flex", alignItems:"center", justifyContent:"center", padding:20 }}>
          <div onClick={e => e.stopPropagation()} style={{ background:C.surface, borderRadius:18, padding:24, width:"100%", maxWidth:380, boxShadow:"0 16px 48px rgba(0,0,0,.2)" }}>
            <h3 style={{ fontFamily:"'Fraunces',serif", fontSize:18, fontWeight:700, color:C.text, marginBottom:6 }}>Link Related Files</h3>
            <p style={{ fontSize:13, color:C.muted, marginBottom:16 }}>The AI will use all linked files together when you ask questions about <strong>{file.name}</strong>.</p>
            <div style={{ display:"flex", flexDirection:"column", gap:8, marginBottom:20 }}>
              {others.map(f => {
                const isLinked = linked.includes(f.id);
                return (
                  <button key={f.id} onClick={() => {
                    const next = isLinked ? linked.filter(id=>id!==f.id) : [...linked, f.id];
                    onSave(next);
                  }}
                    style={{ display:"flex", alignItems:"center", gap:12, padding:"10px 14px", borderRadius:10, border:`1.5px solid ${isLinked?C.accent:C.border}`, background:isLinked?C.accentL:"#fff", cursor:"pointer", textAlign:"left" }}>
                    <span style={{ fontSize:18 }}>{isLinked?"✅":"⬜"}</span>
                    <span style={{ fontSize:13, fontWeight:600, color:isLinked?C.accent:C.text, flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{f.name}</span>
                  </button>
                );
              })}
            </div>
            <button onClick={() => setOpen(false)}
              style={{ width:"100%", background:C.accent, color:"#fff", border:"none", borderRadius:10, padding:"11px", fontSize:14, fontWeight:700, cursor:"pointer" }}>
              Done
            </button>
          </div>
        </div>
      )}
    </>
  );
}


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
      FILE_STORE.set(id, f);
      idbSave(id, f); // persist to IndexedDB so it survives page close
      return {
        id, name: f.name, type: f.type, size: f.size,
        colorIndex: 0, notes: "", studyCards: [], uploadedAt: new Date().toLocaleDateString(),
        linkedFileIds: [], _fileObj: f,
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
                              {(file.linkedFileIds||[]).length > 0 && <span style={{ marginLeft:8, color:C.accent }}>🔗 {(file.linkedFileIds||[]).length} linked</span>}
                            </p>
                          </div>
                          <div style={{ display:"flex", gap:4 }}>
                            {FILE_COLORS.map((col,ci) => (
                              <button key={ci} onClick={() => onUpdate({...folder,files:folder.files.map(f=>f.id===file.id?{...f,colorIndex:ci}:f)})}
                                style={{ width:14, height:14, borderRadius:"50%", background:col.accent, border:file.colorIndex===ci?`2px solid ${C.text}`:"2px solid transparent", cursor:"pointer" }} />
                            ))}
                          </div>
                          <LinkBtn file={file} allFiles={folder.files}
                            onSave={ids => onUpdate({...folder,files:folder.files.map(f=>f.id===file.id?{...f,linkedFileIds:ids}:f)})} />
                          <button onClick={() => onOpenFile(file)} className="hov"
                            style={{ display:"flex", alignItems:"center", gap:6, background:C.accentL, color:C.accent, border:"none", borderRadius:8, padding:"7px 14px", fontSize:13, fontWeight:600, cursor:"pointer" }}>
                            <Icon d={I.edit} size={13} color={C.accent} /> Open
                          </button>
                          <button onClick={() => { idbDelete(file.id); FILE_STORE.delete(file.id); onUpdate({...folder,files:folder.files.filter(f=>f.id!==file.id)}); }} className="hov"
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
    {id:"view",label:"View File",icon:I.file},
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
  const fileObj = file._fileObj || FILE_STORE.get(file.id) || null;
  const fileName = fileObj?.name || file.name || "";
  const ext  = fileName.split(".").pop().toLowerCase();
  const mime = fileObj?.type || "";

  const isPDF   = ext === "pdf" || mime === "application/pdf";
  const isImage = mime.startsWith("image/") || ["jpg","jpeg","png","gif","webp","bmp","svg"].includes(ext);
  const isText  = ["txt","md","csv","json","js","ts","jsx","py","html","css","xml","yaml","yml"].includes(ext);
  const isWord  = ["doc","docx"].includes(ext);
  const isPPT   = ["ppt","pptx"].includes(ext);
  const isExcel = ["xls","xlsx"].includes(ext);

  // PDF state
  const canvasRef  = useRef(null);
  const drawRef    = useRef(null);
  const renderRef  = useRef(null);
  const pdfRef     = useRef(null);
  const [totalPages, setTotalPages] = useState(0);
  const [pageNum,    setPageNum]    = useState(1);
  const [pdfReady,   setPdfReady]   = useState(false);
  const [annotations,setAnnotations]= useState({});
  const [tool,      setTool]      = useState("pen");
  const [penColor,  setPenColor]  = useState("#E53E3E");
  const [brushSize, setBrushSize] = useState(3);
  const [drawing,   setDrawing]   = useState(false);
  const lastPosRef = useRef(null);

  // Explain state
  const [explaining,  setExplaining]  = useState(false);
  const [explanation, setExplanation] = useState("");
  const [showExplain, setShowExplain] = useState(false);

  // PPT page nav
  const [pptTotal, setPptTotal] = useState(0);
  const [pptPage,  setPptPage]  = useState(1);

  // Load PDF
  useEffect(() => {
    if (!isPDF || !fileObj) return;
    setPdfReady(false); pdfRef.current = null; setPageNum(1);
    (async () => {
      try {
        if (!window.pdfjsLib) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
            s.onload = res; s.onerror = rej; document.head.appendChild(s);
          });
          window.pdfjsLib.GlobalWorkerOptions.workerSrc =
            "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
        }
        const buf = await fileObj.arrayBuffer();
        const pdf = await window.pdfjsLib.getDocument({ data: new Uint8Array(buf) }).promise;
        pdfRef.current = pdf;
        setTotalPages(pdf.numPages);
        setPdfReady(true);
      } catch(e) { console.error("PDF load:", e); }
    })();
  }, [fileObj]);

  // Render PDF page
  useEffect(() => {
    if (!pdfReady || !pdfRef.current) return;
    const pdf = pdfRef.current; const pg = pageNum;
    (async () => {
      if (renderRef.current) { try { renderRef.current.cancel(); } catch {} renderRef.current = null; }
      const c = canvasRef.current; if (!c) return;
      try {
        const pdfPage = await pdf.getPage(pg);
        const parentW = c.parentElement?.offsetWidth || 760;
        const base    = pdfPage.getViewport({ scale: 1 });
        const scale   = Math.min(2.5, (parentW - 32) / base.width);
        const vp      = pdfPage.getViewport({ scale });
        c.width = vp.width; c.height = vp.height;
        const ctx = c.getContext("2d");
        ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, c.width, c.height);
        const task = pdfPage.render({ canvasContext: ctx, viewport: vp });
        renderRef.current = task;
        await task.promise; renderRef.current = null;
        const dc = drawRef.current;
        if (dc) {
          dc.width = vp.width; dc.height = vp.height;
          dc.getContext("2d").clearRect(0, 0, dc.width, dc.height);
          if (annotations[pg]) {
            const img = new Image();
            img.onload = () => dc.getContext("2d")?.drawImage(img, 0, 0);
            img.src = annotations[pg];
          }
        }
      } catch(e) { if (e?.name !== "RenderingCancelledException") console.error("Render:", e); }
    })();
  }, [pdfReady, pageNum]);

  const saveAnnotation = () => {
    if (drawRef.current) setAnnotations(p => ({ ...p, [pageNum]: drawRef.current.toDataURL() }));
  };
  const changePage = (n) => {
    const v = Math.max(1, Math.min(totalPages, n));
    if (v === pageNum) return;
    saveAnnotation(); setPageNum(v);
  };
  const getPos = (e) => {
    const dc = drawRef.current;
    const r  = dc.getBoundingClientRect();
    const sx = dc.width / r.width, sy = dc.height / r.height;
    const src = e.touches?.[0] || e;
    return { x: (src.clientX - r.left) * sx, y: (src.clientY - r.top) * sy };
  };
  const startDraw = (e) => { e.preventDefault(); lastPosRef.current = getPos(e); setDrawing(true); };
  const doDraw = (e) => {
    e.preventDefault();
    if (!drawing || !lastPosRef.current || !drawRef.current) return;
    const ctx = drawRef.current.getContext("2d");
    const pos = getPos(e);
    ctx.beginPath(); ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y); ctx.lineTo(pos.x, pos.y);
    if (tool === "eraser")         { ctx.globalCompositeOperation = "destination-out"; ctx.lineWidth = brushSize * 5; }
    else if (tool === "highlight") { ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 0.3; ctx.lineWidth = brushSize * 8; ctx.strokeStyle = penColor; }
    else                           { ctx.globalCompositeOperation = "source-over"; ctx.globalAlpha = 1; ctx.lineWidth = brushSize; ctx.strokeStyle = penColor; }
    ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
    ctx.globalAlpha = 1; ctx.globalCompositeOperation = "source-over";
    lastPosRef.current = pos;
  };
  const stopDraw = () => { if (drawing) { setDrawing(false); saveAnnotation(); } };
  const clearDraw = () => {
    drawRef.current?.getContext("2d").clearRect(0, 0, drawRef.current.width, drawRef.current.height);
    setAnnotations(p => { const n = {...p}; delete n[pageNum]; return n; });
  };

  const doExplain = async () => {
    setExplaining(true); setShowExplain(true); setExplanation("");
    try {
      let text = "";
      if (pdfRef.current) {
        const pg = await pdfRef.current.getPage(pageNum);
        text = (await pg.getTextContent()).items.map(i => i.str).join(" ").trim().slice(0, 3000);
      } else if (fileObj) {
        text = (await extractFileText(fileObj).catch(() => "")).slice(0, 3000);
      }
      const res = await callClaude(
        "You are a helpful study tutor. Explain clearly and simply. Plain text only — no asterisks, no markdown.",
        text
          ? `Explain ONLY this content from page ${pageNum} of "${file.name}":

${text}`
          : `Explain the topic "${file.name}" to a student.`
      );
      setExplanation(res);
    } catch(e) { setExplanation("Error: " + e.message); }
    setExplaining(false);
  };

  const COLORS = ["#E53E3E","#FF8C00","#ECC94B","#38A169","#3182CE","#805AD5","#1a1a1a","#ffffff"];
  const TOOLS  = [{id:"pen",icon:"✏️"},{id:"highlight",icon:"🖊️"},{id:"eraser",icon:"🧹"}];

  if (!fileObj) return (
    <div style={{ textAlign:"center", padding:"60px 24px" }}>
      <div style={{ fontSize:48, marginBottom:12 }}>📂</div>
      <p style={{ fontSize:16, fontWeight:600, color:C.text, marginBottom:8 }}>File not loaded</p>
      <p style={{ fontSize:13, color:C.muted, marginBottom:20 }}>Files need to be re-uploaded once after a full page refresh.</p>
      <label style={{ display:"inline-flex", alignItems:"center", gap:8, background:C.accent, color:"#fff", borderRadius:10, padding:"11px 22px", cursor:"pointer", fontSize:14, fontWeight:600 }}>
        📁 Re-open File
        <input type="file" style={{ display:"none" }} onChange={e => {
          const f = e.target.files?.[0]; if (!f) return;
          FILE_STORE.set(file.id, f); idbSave(file.id, f); onUpdate({...file, _fileObj: f});
        }} />
      </label>
    </div>
  );

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"calc(100vh - 112px)" }}>

      {/* Toolbar */}
      <div style={{ background:C.surface, borderBottom:`1px solid ${C.border}`, padding:"6px 14px", display:"flex", alignItems:"center", gap:8, flexWrap:"wrap" }}>
        {isPDF ? (<>
          {TOOLS.map(t => (
            <button key={t.id} onClick={() => setTool(t.id)}
              style={{ width:32, height:32, borderRadius:7, border:`1.5px solid ${tool===t.id?C.accent:C.border}`, background:tool===t.id?C.accentL:"#fff", cursor:"pointer", fontSize:14 }}>
              {t.icon}
            </button>
          ))}
          <div style={{ width:1, height:20, background:C.border, flexShrink:0 }} />
          {COLORS.map(col => (
            <button key={col} onClick={() => setPenColor(col)}
              style={{ width:18, height:18, borderRadius:"50%", background:col, border:penColor===col?`3px solid ${C.accent}`:`1.5px solid ${C.border}`, cursor:"pointer", flexShrink:0 }} />
          ))}
          <div style={{ width:1, height:20, background:C.border, flexShrink:0 }} />
          <input type="range" min="1" max="20" value={brushSize} onChange={e => setBrushSize(+e.target.value)} style={{ width:60 }} />
          <button onClick={clearDraw} style={{ fontSize:12, background:"none", border:`1px solid ${C.border}`, borderRadius:6, padding:"4px 8px", cursor:"pointer", color:C.muted }}>🗑️ Clear</button>
          <div style={{ flex:1 }} />
          <div style={{ display:"flex", alignItems:"center", gap:5 }}>
            <button onClick={() => changePage(pageNum-1)} disabled={pageNum<=1}
              style={{ width:28, height:28, borderRadius:6, border:`1px solid ${C.border}`, background:"#fff", cursor:pageNum<=1?"default":"pointer", opacity:pageNum<=1?.4:1, fontSize:18 }}>‹</button>
            <input type="number" value={pageNum} min="1" max={totalPages}
              onChange={e => changePage(parseInt(e.target.value)||1)}
              style={{ width:42, textAlign:"center", border:`1px solid ${C.border}`, borderRadius:6, padding:"3px 4px", fontSize:13, outline:"none" }} />
            <span style={{ fontSize:12, color:C.muted }}>/ {totalPages}</span>
            <button onClick={() => changePage(pageNum+1)} disabled={pageNum>=totalPages}
              style={{ width:28, height:28, borderRadius:6, border:`1px solid ${C.border}`, background:"#fff", cursor:pageNum>=totalPages?"default":"pointer", opacity:pageNum>=totalPages?.4:1, fontSize:18 }}>›</button>
          </div>
          <button onClick={doExplain} disabled={explaining}
            style={{ display:"flex", alignItems:"center", gap:5, background:C.accent, color:"#fff", border:"none", borderRadius:7, padding:"6px 14px", fontSize:13, fontWeight:600, cursor:explaining?"default":"pointer", whiteSpace:"nowrap" }}>
            <Icon d={I.sparkle} size={12} color="#fff" sw={2} />{explaining?"…":`Explain Page ${pageNum}`}
          </button>
        </>) : (<>
          <span style={{ fontSize:13, color:C.muted }}>📄 <strong style={{ color:C.text }}>{fileName}</strong></span>
          <div style={{ flex:1 }} />
          {isPPT && pptTotal > 1 && (
            <div style={{ display:"flex", alignItems:"center", gap:5 }}>
              <button onClick={() => setPptPage(p=>Math.max(1,p-1))} disabled={pptPage<=1}
                style={{ width:28, height:28, borderRadius:6, border:`1px solid ${C.border}`, background:"#fff", cursor:pptPage<=1?"default":"pointer", opacity:pptPage<=1?.4:1, fontSize:18 }}>‹</button>
              <span style={{ fontSize:13, color:C.muted }}>{pptPage}/{pptTotal}</span>
              <button onClick={() => setPptPage(p=>Math.min(pptTotal,p+1))} disabled={pptPage>=pptTotal}
                style={{ width:28, height:28, borderRadius:6, border:`1px solid ${C.border}`, background:"#fff", cursor:pptPage>=pptTotal?"default":"pointer", opacity:pptPage>=pptTotal?.4:1, fontSize:18 }}>›</button>
            </div>
          )}
          <button onClick={doExplain} disabled={explaining}
            style={{ display:"flex", alignItems:"center", gap:5, background:C.accent, color:"#fff", border:"none", borderRadius:7, padding:"6px 14px", fontSize:13, fontWeight:600, cursor:explaining?"default":"pointer" }}>
            <Icon d={I.sparkle} size={12} color="#fff" sw={2} />{explaining?"Explaining…":"AI Explain"}
          </button>
        </>)}
      </div>

      {/* Viewer */}
      <div style={{ flex:1, display:"flex", overflow:"hidden" }}>
        <div style={{ flex:1, overflow:"auto", background:"#404040", padding:"24px", display:"flex", justifyContent:"center", alignItems:"flex-start" }}>

          {isPDF && (
            <div style={{ position:"relative", display:"inline-block", lineHeight:0, boxShadow:"0 4px 32px rgba(0,0,0,.6)" }}>
              <canvas ref={canvasRef} style={{ display:"block" }} />
              <canvas ref={drawRef}
                style={{ position:"absolute", top:0, left:0, width:"100%", height:"100%", cursor:tool==="eraser"?"cell":"crosshair", touchAction:"none" }}
                onMouseDown={startDraw} onMouseMove={doDraw} onMouseUp={stopDraw} onMouseLeave={stopDraw}
                onTouchStart={startDraw} onTouchMove={doDraw} onTouchEnd={stopDraw} />
            </div>
          )}

          {isImage  && <ImageViewer  fileObj={fileObj} fileName={fileName} />}
          {isText   && <TextViewer   fileObj={fileObj} />}
          {isWord   && <WordViewer   fileObj={fileObj} />}
          {isPPT    && <PPTViewer    fileObj={fileObj} page={pptPage} onTotalPages={setPptTotal} />}
          {isExcel  && <ExcelViewer  fileObj={fileObj} />}

          {!isPDF && !isImage && !isText && !isWord && !isPPT && !isExcel && (
            <DownloadViewer fileObj={fileObj} fileName={fileName} />
          )}
        </div>

        {/* Explain panel */}
        {showExplain && (
          <div style={{ width:300, background:C.surface, borderLeft:`1px solid ${C.border}`, display:"flex", flexDirection:"column", flexShrink:0 }}>
            <div style={{ padding:"12px 16px", borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", justifyContent:"space-between" }}>
              <span style={{ fontSize:13, fontWeight:700, color:C.text }}>
                {isPDF ? `Page ${pageNum} — Explanation` : "AI Explanation"}
              </span>
              <button onClick={() => setShowExplain(false)} style={{ background:"none", border:"none", cursor:"pointer" }}>
                <Icon d={I.x} size={14} color={C.muted} />
              </button>
            </div>
            <div style={{ flex:1, overflowY:"auto", padding:"14px 16px" }}>
              {explaining
                ? <div style={{ display:"flex", gap:5 }}>{[0,1,2].map(j=><div key={j} style={{ width:7,height:7,borderRadius:"50%",background:C.accent,animation:"bounce 1.2s infinite",animationDelay:`${j*.2}s` }}/>)}</div>
                : <div style={{ fontSize:13, lineHeight:1.7, color:C.text, whiteSpace:"pre-wrap" }}>{explanation}</div>
              }
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── FILE VIEWER HELPERS ──────────────────────────────────────────────────────

function ImageViewer({ fileObj, fileName }) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    const u = URL.createObjectURL(fileObj);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [fileObj]);
  if (!url) return null;
  return (
    <img src={url} alt={fileName}
      style={{ maxWidth:"100%", maxHeight:"calc(100vh - 200px)", borderRadius:6, boxShadow:"0 4px 32px rgba(0,0,0,.5)", background:"#fff" }} />
  );
}

function TextViewer({ fileObj }) {
  const [text, setText] = useState("Loading…");
  useEffect(() => {
    readFileAsText(fileObj)
      .then(t => setText(t || "(empty)"))
      .catch(() => setText("Could not read file."));
  }, [fileObj]);
  return (
    <div style={{ background:"#1e1e2e", color:"#cdd6f4", padding:"24px 28px", borderRadius:8, width:"100%", maxWidth:860, boxShadow:"0 8px 32px rgba(0,0,0,.5)", whiteSpace:"pre-wrap", fontFamily:"monospace", fontSize:13, lineHeight:1.8, minHeight:400, wordBreak:"break-word" }}>
      {text}
    </div>
  );
}

function WordViewer({ fileObj }) {
  const [html,    setHtml]    = useState("");
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");
  useEffect(() => {
    (async () => {
      try {
        if (!window.mammoth) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/mammoth/1.6.0/mammoth.browser.min.js";
            s.onload = res; s.onerror = rej; document.head.appendChild(s);
          });
        }
        const ab  = await fileObj.arrayBuffer();
        const out = await window.mammoth.convertToHtml({ arrayBuffer: ab });
        if (out.value) setHtml(out.value);
        else setError("Document appears empty.");
      } catch(e) {
        console.error("Word:", e);
        setError("Could not render this Word file: " + e.message);
      }
      setLoading(false);
    })();
  }, [fileObj]);

  return (
    <div style={{ background:"#fff", padding:"48px 60px", borderRadius:4, maxWidth:860, width:"100%", boxShadow:"0 8px 32px rgba(0,0,0,.4)", minHeight:500, fontSize:14, lineHeight:1.9, color:"#111" }}>
      {loading && <p style={{ color:"#888" }}>Loading document…</p>}
      {error   && <p style={{ color:"#e53e3e" }}>{error}</p>}
      {html    && <div dangerouslySetInnerHTML={{ __html: html }} />}
    </div>
  );
}

function PPTViewer({ fileObj, page, onTotalPages }) {
  const [slides,  setSlides]  = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  useEffect(() => {
    (async () => {
      try {
        if (!window.JSZip) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
            s.onload = res; s.onerror = rej; document.head.appendChild(s);
          });
        }
        const ab  = await fileObj.arrayBuffer();
        const zip = await window.JSZip.loadAsync(ab);

        // Find slides in correct order
        const slideKeys = Object.keys(zip.files)
          .filter(k => /^ppt\/slides\/slide\d+\.xml$/.test(k))
          .sort((a, b) => {
            const na = parseInt(a.match(/\d+/)[0]);
            const nb = parseInt(b.match(/\d+/)[0]);
            return na - nb;
          });

        if (slideKeys.length === 0) { setError("No slides found in this file."); setLoading(false); return; }
        onTotalPages && onTotalPages(slideKeys.length);

        const parsed = await Promise.all(slideKeys.map(async sk => {
          const xml  = await zip.files[sk].async("string");
          // Extract all text nodes
          const texts = [];
          const re = /<a:t[^>]*>([^<]*)<\/a:t>/g;
          let m;
          while ((m = re.exec(xml)) !== null) if (m[1].trim()) texts.push(m[1].trim());
          // Extract embedded images
          const relKey = sk.replace("slides/slide", "slides/_rels/slide").replace(".xml", ".xml.rels");
          const imgs = [];
          if (zip.files[relKey]) {
            const relXml = await zip.files[relKey].async("string");
            const imgRe  = /Target="\.\.\/media\/([^"]+)"/g;
            let rm;
            while ((rm = imgRe.exec(relXml)) !== null) {
              const path = "ppt/media/" + rm[1];
              if (zip.files[path]) {
                const ext2 = rm[1].split(".").pop().toLowerCase();
                const mt   = ext2==="png"?"image/png":ext2==="gif"?"image/gif":ext2==="svg"?"image/svg+xml":"image/jpeg";
                const b64  = await zip.files[path].async("base64");
                imgs.push({ src:`data:${mt};base64,${b64}`, ext: ext2 });
              }
            }
          }
          return { texts, imgs };
        }));

        setSlides(parsed);
      } catch(e) {
        console.error("PPT:", e);
        setError("Could not render PowerPoint: " + e.message);
      }
      setLoading(false);
    })();
  }, [fileObj]);

  const slide = slides[Math.max(0, (page||1) - 1)] || { texts:[], imgs:[] };

  if (loading) return <div style={{ color:"#fff", padding:40 }}>Loading presentation…</div>;
  if (error)   return <div style={{ color:"#fca5a5", padding:40, background:"#1e1e2e", borderRadius:8 }}>{error}</div>;

  return (
    <div style={{ background:"#fff", borderRadius:8, width:"100%", maxWidth:900, minHeight:480, boxShadow:"0 8px 32px rgba(0,0,0,.4)", overflow:"hidden" }}>
      {/* Slide header */}
      {slide.texts[0] && (
        <div style={{ background:"linear-gradient(135deg, #1a1a2e, #16213e)", padding:"28px 36px" }}>
          <p style={{ fontSize:24, fontWeight:800, color:"#fff", lineHeight:1.3 }}>{slide.texts[0]}</p>
        </div>
      )}
      {/* Images */}
      {slide.imgs.length > 0 && (
        <div style={{ padding:"20px 36px 0", display:"flex", gap:12, flexWrap:"wrap" }}>
          {slide.imgs.map((img, i) => (
            <img key={i} src={img.src} alt="" style={{ maxWidth:"100%", maxHeight:280, objectFit:"contain", borderRadius:6, border:"1px solid #e5e7eb" }} />
          ))}
        </div>
      )}
      {/* Body text */}
      <div style={{ padding:"20px 36px 32px" }}>
        {slide.texts.slice(1).map((t, i) => (
          <p key={i} style={{ fontSize:15, color:"#374151", marginBottom:8, lineHeight:1.7 }}>• {t}</p>
        ))}
        {slide.texts.length === 0 && slide.imgs.length === 0 && (
          <p style={{ color:"#9ca3af", fontStyle:"italic", marginTop:40, textAlign:"center" }}>No content on this slide.</p>
        )}
      </div>
    </div>
  );
}

function ExcelViewer({ fileObj }) {
  const [sheets,      setSheets]      = useState([]);
  const [activeSheet, setActiveSheet] = useState(0);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState("");

  useEffect(() => {
    (async () => {
      try {
        if (!window.XLSX) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js";
            s.onload = res; s.onerror = rej; document.head.appendChild(s);
          });
        }
        const ab = await fileObj.arrayBuffer();
        const wb = window.XLSX.read(ab, { type:"array" });
        const all = wb.SheetNames.map(sn => ({
          name: sn,
          rows: window.XLSX.utils.sheet_to_json(wb.Sheets[sn], { header:1, defval:"" }),
        }));
        if (all.length === 0) setError("No sheets found.");
        else setSheets(all);
      } catch(e) {
        console.error("Excel:", e);
        setError("Could not read spreadsheet: " + e.message);
      }
      setLoading(false);
    })();
  }, [fileObj]);

  if (loading) return <div style={{ color:"#fff", padding:40 }}>Loading spreadsheet…</div>;
  if (error)   return <div style={{ color:"#fca5a5", padding:40, background:"#1e1e2e", borderRadius:8 }}>{error}</div>;

  const sheet = sheets[activeSheet] || { rows:[] };
  // Find max columns
  const maxCols = Math.max(...sheet.rows.map(r => r.length), 0);

  return (
    <div style={{ background:"#fff", borderRadius:8, width:"100%", boxShadow:"0 8px 32px rgba(0,0,0,.4)", overflow:"hidden" }}>
      {sheets.length > 1 && (
        <div style={{ display:"flex", background:"#f3f4f6", borderBottom:"1px solid #e5e7eb", overflowX:"auto" }}>
          {sheets.map((s, i) => (
            <button key={i} onClick={() => setActiveSheet(i)}
              style={{ padding:"8px 18px", border:"none", background:activeSheet===i?"#fff":"transparent", borderBottom:activeSheet===i?`2px solid ${C.accent}`:"2px solid transparent", fontWeight:activeSheet===i?700:400, color:activeSheet===i?C.accent:C.muted, fontSize:13, cursor:"pointer", whiteSpace:"nowrap" }}>
              {s.name}
            </button>
          ))}
        </div>
      )}
      <div style={{ overflowX:"auto", maxHeight:"calc(100vh - 250px)" }}>
        <table style={{ borderCollapse:"collapse", width:"100%", fontSize:13, minWidth: maxCols * 100 }}>
          <tbody>
            {sheet.rows.map((row, ri) => (
              <tr key={ri} style={{ background: ri===0?"#eff6ff": ri%2===0?"#fff":"#f9fafb" }}>
                {Array.from({ length: Math.max(row.length, maxCols) }, (_, ci) => (
                  <td key={ci} style={{ border:"1px solid #e5e7eb", padding:"6px 12px", fontWeight:ri===0?700:400, whiteSpace:"nowrap", maxWidth:240, overflow:"hidden", textOverflow:"ellipsis", color: ri===0?"#1d4ed8":"#111" }}>
                    {row[ci] ?? ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DownloadViewer({ fileObj, fileName }) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    const u = URL.createObjectURL(fileObj);
    setUrl(u); return () => URL.revokeObjectURL(u);
  }, [fileObj]);
  const sizeKB = Math.round((fileObj?.size||0)/1024);
  return (
    <div style={{ textAlign:"center", color:"#fff", padding:"60px 20px" }}>
      <div style={{ fontSize:72, marginBottom:20 }}>📎</div>
      <p style={{ fontSize:20, fontWeight:700, marginBottom:8 }}>{fileName}</p>
      <p style={{ opacity:.6, fontSize:13, marginBottom:28 }}>{sizeKB} KB · Preview not available in browser</p>
      {url && <a href={url} download={fileName}
        style={{ display:"inline-flex", alignItems:"center", gap:8, background:C.accent, color:"#fff", borderRadius:10, padding:"12px 24px", fontSize:15, fontWeight:600, textDecoration:"none" }}>
        ⬇️ Download File
      </a>}
    </div>
  );
}


function AITab({ file, allFiles, folder, onUpdate }) {
  const [msgs, setMsgs] = useState([]);
  const [inp, setInp] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedFileIds, setSelectedFileIds] = useState([]);
  const bottomRef = useRef(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);
  const send = async () => {
    const text = inp.trim();
    if (!text || loading) return;
    const userMsg = { role:"user", content: text };
    const newMsgs = [...msgs, userMsg];
    setMsgs(newMsgs); setInp(""); setLoading(true);
    try {
      // Build context from selected files (folder mode) or current file + linked
      let fileContext = "";
      const safeText = async (fObj) => {
        if (!fObj) return "";
        try { const t = await extractFileText(fObj); return (t || "").slice(0, 2000); } catch { return ""; }
      };
      if (selectedFileIds.length > 0) {
        // Folder AI — use whichever files the user selected from dropdown
        for (const sid of selectedFileIds) {
          const sf = allFiles?.find(f => f.id === sid);
          if (sf) { const t = await safeText(sf._fileObj); if (t) fileContext += `File "${sf.name}":
${t}

`; }
        }
      } else if (file) {
        // File AI — use this file + its linked files
        const t = await safeText(file._fileObj);
        if (t) fileContext += `File "${file.name}":
${t}

`;
        for (const lid of (file.linkedFileIds || [])) {
          const lf = allFiles?.find(f => f.id === lid);
          if (lf) { const lt = await safeText(lf._fileObj); if (lt) fileContext += `Linked file "${lf.name}":
${lt}

`; }
        }
      }
      const sys = fileContext
        ? `You are a study AI. Use ONLY the following file content to answer. Plain text, no asterisks, no markdown.

${fileContext}`
        : `You are helping a student with "${folder?.name || "their files"}". Plain text, no asterisks.`;
      const reply = await callClaudeChat(sys, newMsgs.map(m => ({ role:m.role, content:m.content })));
      setMsgs([...newMsgs, { role:"assistant", content: reply }]);
    } catch(e) { setMsgs([...newMsgs, { role:"assistant", content:"Error: " + e.message }]); }
    setLoading(false);
  };
  return (
    <div style={{ display:"flex", flexDirection:"column", height:"calc(100vh - 200px)", minHeight:400 }}>
      <div style={{ flex:1, overflowY:"auto", padding:"16px 0", display:"flex", flexDirection:"column", gap:12 }}>
        {/* Folder-level file selector — smart linked pairs */}
        {!file && allFiles && allFiles.length > 0 && (() => {
          // Build linked groups: each file that has linkedFileIds forms a group
          const linked = allFiles.filter(f => (f.linkedFileIds||[]).length > 0);
          const standalone = allFiles.filter(f => (f.linkedFileIds||[]).length === 0 &&
            !allFiles.some(o => (o.linkedFileIds||[]).includes(f.id)));
          // Deduplicate groups: group = [file, ...its linked files]
          const seen = new Set();
          const groups = [];
          for (const f of linked) {
            if (seen.has(f.id)) continue;
            const members = [f, ...(f.linkedFileIds||[]).map(id => allFiles.find(x=>x.id===id)).filter(Boolean)];
            members.forEach(m => seen.add(m.id));
            groups.push(members);
          }
          return (
            <div style={{ background:C.surface, border:`1px solid ${C.border}`, borderRadius:16, padding:"14px 16px", marginBottom:10 }}>
              <p style={{ fontSize:11, fontWeight:800, color:C.muted, letterSpacing:.8, marginBottom:12 }}>CHOOSE FILES FOR AI</p>

              {groups.length > 0 && <>
                <p style={{ fontSize:11, fontWeight:700, color:C.accent, marginBottom:8 }}>🔗 LINKED PAIRS</p>
                {groups.map((grp, gi) => {
                  const allSel = grp.every(f => selectedFileIds.includes(f.id));
                  const someSel = grp.some(f => selectedFileIds.includes(f.id));
                  return (
                    <div key={gi} style={{ border:`2px solid ${allSel?C.accent:someSel?C.accentS:C.border}`, borderRadius:12, padding:"10px 12px", marginBottom:8, background:allSel?C.accentL:someSel?"#f0f5ff":"#fff" }}>
                      <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:6 }}>
                        <span style={{ fontSize:11, fontWeight:700, color:C.accent }}>Group {gi+1}</span>
                        <button onClick={() => {
                          const ids = grp.map(f=>f.id);
                          setSelectedFileIds(prev => allSel ? prev.filter(id=>!ids.includes(id)) : [...new Set([...prev,...ids])]);
                        }} style={{ fontSize:11, fontWeight:700, background:allSel?C.accent:C.accentL, color:allSel?"#fff":C.accent, border:"none", borderRadius:20, padding:"3px 10px", cursor:"pointer" }}>
                          {allSel?"Deselect all":"Select all"}
                        </button>
                      </div>
                      {grp.map(f => {
                        const sel = selectedFileIds.includes(f.id);
                        return (
                          <button key={f.id} onClick={() => setSelectedFileIds(prev => sel ? prev.filter(id=>id!==f.id) : [...prev, f.id])}
                            style={{ display:"flex", alignItems:"center", gap:8, width:"100%", padding:"6px 8px", borderRadius:8, border:`1.5px solid ${sel?C.accent:C.border}`, background:sel?C.accentL:"#f8f8f8", cursor:"pointer", textAlign:"left", marginBottom:4 }}>
                            <span>{sel?"✅":"⬜"}</span>
                            <span style={{ fontSize:12, fontWeight:600, color:sel?C.accent:C.text, flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{f.name}</span>
                          </button>
                        );
                      })}
                    </div>
                  );
                })}
              </>}

              {standalone.length > 0 && <>
                <p style={{ fontSize:11, fontWeight:700, color:C.muted, marginBottom:8, marginTop: groups.length>0?8:0 }}>📄 INDIVIDUAL FILES</p>
                {standalone.map(f => {
                  const sel = selectedFileIds.includes(f.id);
                  return (
                    <button key={f.id} onClick={() => setSelectedFileIds(prev => sel ? prev.filter(id=>id!==f.id) : [...prev, f.id])}
                      style={{ display:"flex", alignItems:"center", gap:8, width:"100%", padding:"7px 10px", borderRadius:9, border:`1.5px solid ${sel?C.accent:C.border}`, background:sel?C.accentL:"#fff", cursor:"pointer", textAlign:"left", marginBottom:5 }}>
                      <span>{sel?"✅":"⬜"}</span>
                      <span style={{ fontSize:13, fontWeight:600, color:sel?C.accent:C.text, flex:1, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{f.name}</span>
                    </button>
                  );
                })}
              </>}

              {selectedFileIds.length > 0 && (
                <p style={{ fontSize:11, color:C.accent, marginTop:8, fontWeight:700 }}>✓ AI will read {selectedFileIds.length} file{selectedFileIds.length>1?"s":""}</p>
              )}
            </div>
          );
        })()}
        {msgs.length === 0 && (
          <div style={{ textAlign:"center", padding:"24px 20px", color:C.muted }}>
            <div style={{ fontSize:40, marginBottom:12 }}>🤖</div>
            <p style={{ fontSize:15, fontWeight:600, color:C.text, marginBottom:6 }}>AI Assistant</p>
            <p style={{ fontSize:13 }}>{file ? `Ask anything about "${file.name}".` : selectedFileIds.length > 0 ? "Ask anything about the selected files." : "Select files above then ask a question."}</p>
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
      const styleGuide = {
        detailed: "Write detailed notes split into sections. Each section has a heading in ALL CAPS followed by bullet points.",
        bullet: "Write ONLY bullet points grouped under ALL CAPS headings. One fact per line.",
        simple: "Write very simple short notes in plain English. Short sentences. No complex words.",
        exam: "Write exam revision notes. Include key terms, definitions, possible exam questions, and a checklist at the end.",
      };
      const txt = await callClaude(
        `You are a study notes writer. ${styleGuide[noteStyle] || styleGuide.detailed}

STRICT FORMATTING RULES - you MUST follow these exactly:
1. NEVER use asterisks (*) or double asterisks (**) anywhere - not even once
2. NEVER use pound signs (#) for headings
3. Write section headings in ALL CAPS on their own line like: INTRODUCTION
4. Use a dash (-) for bullet points
5. Use plain text only - no markdown, no symbols except dashes and dots
6. ONLY use content from the file if provided - do not add outside facts`,
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
    recognition.maxAlternatives = 3;
    let interimText = "";
    recognition.onresult = (e) => {
      interimText = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) {
          // Pick the most confident result
          let best = e.results[i][0].transcript;
          let bestConf = e.results[i][0].confidence || 0;
          for (let a = 1; a < e.results[i].length; a++) {
            if ((e.results[i][a].confidence || 0) > bestConf) {
              best = e.results[i][a].transcript;
              bestConf = e.results[i][a].confidence || 0;
            }
          }
          transcriptRef.current += best + " ";
        } else {
          interimText += e.results[i][0].transcript;
        }
      }
      const preview = (transcriptRef.current + interimText).trim();
      setVoiceStatus("🎙️ " + (preview ? preview.slice(-80) + "…" : "Listening… speak now"));
    };
    recognition.onerror = (e) => {
      if (e.error === "not-allowed") setVoiceStatus("❌ Microphone access denied. Please allow mic in browser settings.");
      else if (e.error === "audio-capture") setVoiceStatus("❌ No microphone found.");
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
    setVoiceStatus("🎙️ Listening… speak now. Click Stop when done.");
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
      // Get context from existing notes and file name to fix pronunciation errors
      const context = `File: "${file.name}". Existing notes context: ${(notes || "").slice(0, 400) || "none"}`;

      const result = await callClaude(
        `You are an expert note-taker. A student recorded a voice note from a lecture or study session.
You have context about the topic to help fix pronunciation and speech recognition errors.

STRICT RULES:
1. Use the topic context to fix likely speech recognition errors (e.g. "Assam" near science context likely means "atom", mispronounced subject terms should be corrected to their proper spelling)
2. Fix grammar, remove all filler words (um, uh, like, you know, sort of, kind of)
3. ONLY use what was said — never add outside information beyond fixing errors
4. Write section headings in ALL CAPS on their own line
5. Use a dash (-) for every bullet point
6. NEVER use asterisks (*) or double asterisks (**) anywhere
7. NEVER use pound signs (#) or any markdown formatting
8. Plain text only — no symbols except dashes
9. Make notes clear and useful for exam revision

Topic context: ${context}`,
        `Fix pronunciation errors using the topic context, then turn this into clean study notes:

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
  if (activeGame==="fillblank") return <FillBlank cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="flashcard") return <FlashcardFlip cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="quizshow") return <QuizShow cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="rapidfire") return <RapidFire cards={cards} onBack={()=>setActiveGame(null)} />;
  if (activeGame==="voice") return <VoiceAnswer cards={cards} onBack={()=>setActiveGame(null)} />;

  const GAMES = [
    {id:"mcq",emoji:"🧠",title:"Multiple Choice",desc:"4 options — pick the right one",bg:C.accentL,accent:C.accent},
    {id:"voice",emoji:"🎙️",title:"Voice Answer",desc:"Speak your answer out loud — AI grades it",bg:"#f5f3ff",accent:"#7c3aed"},
    {id:"flashcard",emoji:"🃏",title:"Flashcard Flip",desc:"Flip cards and track what you know",bg:"#f5f3ff",accent:"#7c3aed"},
    {id:"quizshow",emoji:"🎤",title:"Quiz Show",desc:"Who Wants to Be a Millionaire style with lifelines",bg:"#fef2f2",accent:"#dc2626"},
    {id:"fillblank",emoji:"✏️",title:"Fill in the Blank",desc:"Complete the sentence with the right answer",bg:"#ecfeff",accent:"#0694a2"},
    {id:"rapidfire",emoji:"⚡",title:"Rapid Fire",desc:"Type as many correct answers as you can in 45s",bg:"#f0fdf4",accent:"#059669"},
    {id:"truefalse",emoji:"✅",title:"True or False",desc:"Decide if the statement is true or false",bg:"#FAF5FF",accent:"#6B46C1"},
    {id:"memory",emoji:"🎴",title:"Memory Flip",desc:"Match question cards to answer cards",bg:"#FFF5F5",accent:"#C53030"},
    {id:"match",emoji:"🔗",title:"Matching Pairs",desc:"Connect each term to its definition",bg:C.greenL,accent:C.green},
    {id:"scramble",emoji:"🔀",title:"Word Scramble",desc:"Unscramble the answer letters",bg:C.warmL,accent:C.warm},
    {id:"speedrun",emoji:"🏃",title:"Speed Run",desc:"Answer as many as you can in 60 seconds",bg:"#FFFFF0",accent:"#D69E2E"},
    {id:"tower",emoji:"🏗️",title:"Answer Tower",desc:"Build a tower — answer correctly to stack blocks",bg:"#E6FFFA",accent:"#2C7A7B"},
    {id:"falling",emoji:"🧱",title:"Falling Blocks",desc:"Type the answer before the block falls",bg:C.purpleL,accent:C.purple},
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


// ─── GAME: FILL IN THE BLANK ──────────────────────────────────────────────────
function FillBlank({ cards, onBack }) {
  const accent = "#0694a2";
  const [deck] = useState(() => [...cards].sort(() => Math.random() - .5));
  const [curr, setCurr] = useState(0);
  const [inp, setInp] = useState("");
  const [result, setResult] = useState(null);
  const [score, setScore] = useState(0);
  const [done, setDone] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => { if (inputRef.current) inputRef.current.focus(); }, [curr]);

  const card = deck[curr];
  // Hide last word of question to create blank
  const words = (card?.question || "").trim().split(" ");
  const blank = words.length > 1 ? words[words.length - 1].replace(/[?:.]/g, "") : card?.answer;
  const questionWithBlank = words.length > 1 ? words.slice(0, -1).join(" ") + " ___?" : "What is the answer?";

  const check = () => {
    if (!inp.trim()) return;
    const ok = inp.trim().toLowerCase() === (card.answer || "").trim().toLowerCase() ||
               inp.trim().toLowerCase() === blank.toLowerCase();
    setResult(ok);
    if (ok) setScore(s => s + 1);
  };

  const next = () => {
    if (curr + 1 >= deck.length) { setDone(true); return; }
    setCurr(c => c + 1); setInp(""); setResult(null);
  };

  if (done) return <GResults score={score} total={deck.length} onBack={onBack} msg="Fill in the Blank done!" />;

  return (
    <div style={{ maxWidth: 540, margin: "0 auto" }}>
      <GHeader title="Fill in the Blank" score={score} curr={curr} total={deck.length} onBack={onBack} accent={accent} />
      <div style={{ background: "#E6FFFE", border: `2px solid ${accent}33`, borderRadius: 18, padding: "30px 28px", marginBottom: 20, textAlign: "center" }}>
        <p style={{ fontSize: 11, fontWeight: 700, color: accent, letterSpacing: 1, marginBottom: 14, textTransform: "uppercase" }}>Complete the sentence</p>
        <p style={{ fontSize: 20, fontWeight: 600, color: C.text, lineHeight: 1.5 }}>{questionWithBlank}</p>
      </div>
      {result === null ? (
        <div style={{ display: "flex", gap: 10 }}>
          <input ref={inputRef} value={inp} onChange={e => setInp(e.target.value)}
            onKeyDown={e => e.key === "Enter" && check()}
            placeholder="Type your answer…"
            style={{ flex: 1, border: `2px solid ${C.border}`, borderRadius: 12, padding: "12px 16px", fontSize: 15, outline: "none", color: C.text }} />
          <button onClick={check} disabled={!inp.trim()}
            style={{ background: accent, color: "#fff", border: "none", borderRadius: 12, padding: "12px 22px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>Check</button>
        </div>
      ) : (
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 44, marginBottom: 10 }}>{result ? "✅" : "❌"}</div>
          {!result && <p style={{ color: C.red, fontSize: 15, marginBottom: 6 }}>Correct answer: <strong>{card.answer}</strong></p>}
          <button onClick={next} style={{ background: accent, color: "#fff", border: "none", borderRadius: 12, padding: "12px 32px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>
            {curr + 1 >= deck.length ? "See Results" : "Next →"}
          </button>
        </div>
      )}
    </div>
  );
}

// ─── GAME: FLASHCARD FLIP ─────────────────────────────────────────────────────
function FlashcardFlip({ cards, onBack }) {
  const accent = "#7c3aed";
  const [deck] = useState(() => [...cards].sort(() => Math.random() - .5));
  const [curr, setCurr] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState(0);
  const [reviewing, setReviewing] = useState([]);
  const [done, setDone] = useState(false);

  const card = deck[curr];

  const respond = (didKnow) => {
    if (!didKnow) setReviewing(r => [...r, card]);
    else setKnown(k => k + 1);
    if (curr + 1 >= deck.length) { setDone(true); return; }
    setCurr(c => c + 1); setFlipped(false);
  };

  if (done) return (
    <div style={{ maxWidth: 480, margin: "0 auto", textAlign: "center", padding: "40px 0" }}>
      <div style={{ fontSize: 56, marginBottom: 12 }}>{known >= deck.length * .8 ? "🌟" : "📚"}</div>
      <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 26, fontWeight: 700, color: C.text, marginBottom: 8 }}>Done!</h2>
      <p style={{ fontSize: 32, fontWeight: 700, color: accent, marginBottom: 4 }}>{known}/{deck.length} known</p>
      <p style={{ fontSize: 14, color: C.muted, marginBottom: 28 }}>{reviewing.length > 0 ? `${reviewing.length} cards to review again` : "You knew them all!"}</p>
      <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
        {reviewing.length > 0 && <button onClick={() => { /* restart with reviewing cards */ }} style={{ background: "#FFF5F5", color: C.red, border: `1.5px solid ${C.red}33`, borderRadius: 12, padding: "11px 22px", fontSize: 14, fontWeight: 700, cursor: "pointer" }}>Review {reviewing.length} missed</button>}
        <button onClick={onBack} style={{ background: accent, color: "#fff", border: "none", borderRadius: 12, padding: "11px 22px", fontSize: 14, fontWeight: 700, cursor: "pointer" }}>← Back</button>
      </div>
    </div>
  );

  return (
    <div style={{ maxWidth: 540, margin: "0 auto" }}>
      <GHeader title="Flashcard Flip" score={known} curr={curr} total={deck.length} onBack={onBack} accent={accent} />
      <div onClick={() => setFlipped(f => !f)} style={{ cursor: "pointer", perspective: 900, height: 240, marginBottom: 20 }}>
        <div style={{ position: "relative", width: "100%", height: "100%", transformStyle: "preserve-3d", transition: "transform .5s", transform: flipped ? "rotateY(180deg)" : "rotateY(0deg)" }}>
          <div style={{ position: "absolute", inset: 0, backfaceVisibility: "hidden", WebkitBackfaceVisibility: "hidden", background: `linear-gradient(135deg, #ede9fe, #ddd6fe)`, border: `2px solid ${accent}33`, borderRadius: 20, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "24px 28px", textAlign: "center" }}>
            <p style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: 1, marginBottom: 12 }}>QUESTION — tap to flip</p>
            <p style={{ fontSize: 19, fontWeight: 600, color: C.text, lineHeight: 1.5 }}>{card.question}</p>
          </div>
          <div style={{ position: "absolute", inset: 0, backfaceVisibility: "hidden", WebkitBackfaceVisibility: "hidden", transform: "rotateY(180deg)", background: `linear-gradient(135deg, #f0fdf4, #dcfce7)`, border: "2px solid #16a34a33", borderRadius: 20, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "24px 28px", textAlign: "center" }}>
            <p style={{ fontSize: 12, fontWeight: 700, color: "#16a34a", letterSpacing: 1, marginBottom: 12 }}>ANSWER</p>
            <p style={{ fontSize: 19, fontWeight: 600, color: C.text, lineHeight: 1.5 }}>{card.answer}</p>
          </div>
        </div>
      </div>
      {flipped && (
        <div style={{ display: "flex", gap: 12 }}>
          <button onClick={() => respond(false)} style={{ flex: 1, background: "#FFF5F5", color: C.red, border: `2px solid ${C.red}33`, borderRadius: 14, padding: "14px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>😕 Still learning</button>
          <button onClick={() => respond(true)} style={{ flex: 1, background: "#f0fdf4", color: "#16a34a", border: "2px solid #16a34a33", borderRadius: 14, padding: "14px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>✅ Got it!</button>
        </div>
      )}
      {!flipped && <p style={{ textAlign: "center", color: C.muted, fontSize: 13 }}>Tap the card to reveal the answer</p>}
    </div>
  );
}

// ─── GAME: QUIZ SHOW ──────────────────────────────────────────────────────────
function QuizShow({ cards, onBack }) {
  const accent = "#dc2626";
  const [deck] = useState(() => [...cards].sort(() => Math.random() - .5));
  const [curr, setCurr] = useState(0);
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [sel, setSel] = useState(null);
  const [done, setDone] = useState(false);
  const [lifelines, setLifelines] = useState({ fifty: true, skip: true });

  const card = deck[curr];
  const [opts] = useState(() => deck.map(c => {
    const wrong = cards.filter(x => x.answer !== c.answer).sort(() => Math.random() - .5).slice(0, 3).map(x => x.answer);
    return [...wrong, c.answer].sort(() => Math.random() - .5);
  }));

  const [visibleOpts, setVisibleOpts] = useState(() => opts[0]);
  useEffect(() => { setVisibleOpts(opts[curr]); }, [curr]);

  const choose = (opt) => {
    if (sel !== null) return;
    setSel(opt);
    if (opt === card.answer) { setScore(s => s + (streak >= 2 ? 2 : 1)); setStreak(s => s + 1); }
    else setStreak(0);
    setTimeout(() => {
      if (curr + 1 >= deck.length) setDone(true);
      else { setCurr(c => c + 1); setSel(null); }
    }, 1200);
  };

  const useFifty = () => {
    if (!lifelines.fifty) return;
    const wrong = visibleOpts.filter(o => o !== card.answer);
    const remove = wrong.sort(() => Math.random() - .5).slice(0, 2);
    setVisibleOpts(v => v.filter(o => !remove.includes(o)));
    setLifelines(l => ({ ...l, fifty: false }));
  };

  const useSkip = () => {
    if (!lifelines.skip) return;
    setLifelines(l => ({ ...l, skip: false }));
    if (curr + 1 >= deck.length) setDone(true);
    else { setCurr(c => c + 1); setSel(null); }
  };

  if (done) return <GResults score={score} total={deck.length} onBack={onBack} msg="Quiz Show Complete! 🎤" />;

  const OPTION_LABELS = ["A", "B", "C", "D"];

  return (
    <div style={{ maxWidth: 580, margin: "0 auto" }}>
      <GHeader title="Quiz Show" score={score} curr={curr} total={deck.length} onBack={onBack} accent={accent} />
      {streak >= 3 && <div style={{ background: "#fef9c3", border: "1.5px solid #ca8a04", borderRadius: 10, padding: "6px 14px", marginBottom: 12, fontSize: 13, fontWeight: 700, color: "#92400e", textAlign: "center" }}>🔥 {streak} in a row! Double points!</div>}
      <div style={{ background: "linear-gradient(135deg,#fef2f2,#fee2e2)", border: `2px solid ${accent}22`, borderRadius: 18, padding: "26px 24px", marginBottom: 20, textAlign: "center" }}>
        <p style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: 1, marginBottom: 12 }}>QUESTION {curr + 1}</p>
        <p style={{ fontSize: 19, fontWeight: 600, color: C.text, lineHeight: 1.5 }}>{card.question}</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
        {visibleOpts.map((opt, i) => {
          let bg = "#fff", border = `1.5px solid ${C.border}`, color = C.text;
          if (sel !== null) {
            if (opt === card.answer) { bg = "#f0fdf4"; border = "2px solid #16a34a"; color = "#16a34a"; }
            else if (opt === sel) { bg = "#fef2f2"; border = `2px solid ${accent}`; color = accent; }
          }
          return (
            <button key={opt} onClick={() => choose(opt)} disabled={sel !== null}
              style={{ background: bg, border, borderRadius: 12, padding: "14px 12px", fontSize: 14, fontWeight: 600, color, cursor: sel !== null ? "default" : "pointer", textAlign: "left", display: "flex", gap: 10, alignItems: "center", transition: "all .15s" }}>
              <span style={{ width: 24, height: 24, borderRadius: "50%", background: accent + "22", color: accent, fontSize: 12, fontWeight: 800, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>{OPTION_LABELS[i]}</span>
              {opt}
            </button>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
        <button onClick={useFifty} disabled={!lifelines.fifty || sel !== null}
          style={{ background: lifelines.fifty ? "#fef9c3" : "#f3f4f6", color: lifelines.fifty ? "#92400e" : C.muted, border: "none", borderRadius: 10, padding: "8px 16px", fontSize: 13, fontWeight: 600, cursor: lifelines.fifty ? "pointer" : "not-allowed" }}>
          50/50 {!lifelines.fifty ? "(used)" : ""}
        </button>
        <button onClick={useSkip} disabled={!lifelines.skip || sel !== null}
          style={{ background: lifelines.skip ? "#eff6ff" : "#f3f4f6", color: lifelines.skip ? "#1d4ed8" : C.muted, border: "none", borderRadius: 10, padding: "8px 16px", fontSize: 13, fontWeight: 600, cursor: lifelines.skip ? "pointer" : "not-allowed" }}>
          Skip {!lifelines.skip ? "(used)" : ""}
        </button>
      </div>
    </div>
  );
}

// ─── GAME: RAPID FIRE ─────────────────────────────────────────────────────────
function RapidFire({ cards, onBack }) {
  const accent = "#059669";
  const TOTAL_TIME = 45;
  const [deck] = useState(() => [...cards].sort(() => Math.random() - .5));
  const [curr, setCurr] = useState(0);
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(TOTAL_TIME);
  const [inp, setInp] = useState("");
  const [flash, setFlash] = useState(null);
  const [done, setDone] = useState(false);
  const inputRef = useRef(null);
  const timerRef = useRef(null);

  useEffect(() => {
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) { clearInterval(timerRef.current); setDone(true); return 0; }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(timerRef.current);
  }, []);

  useEffect(() => { if (inputRef.current) inputRef.current.focus(); }, [curr]);

  const submit = () => {
    const card = deck[curr % deck.length];
    const ok = inp.trim().toLowerCase() === (card.answer || "").trim().toLowerCase();
    setFlash(ok ? "correct" : "wrong");
    if (ok) setScore(s => s + 1);
    setTimeout(() => { setFlash(null); setCurr(c => c + 1); setInp(""); }, 400);
  };

  if (done) return (
    <div style={{ maxWidth: 420, margin: "0 auto", textAlign: "center", padding: "40px 0" }}>
      <div style={{ fontSize: 56, marginBottom: 12 }}>⚡</div>
      <h2 style={{ fontFamily: "'Fraunces',serif", fontSize: 28, fontWeight: 700, color: C.text, marginBottom: 8 }}>Time Up!</h2>
      <p style={{ fontSize: 48, fontWeight: 700, color: accent, marginBottom: 4 }}>{score}</p>
      <p style={{ fontSize: 16, color: C.muted, marginBottom: 28 }}>correct answers in {TOTAL_TIME}s</p>
      <button onClick={onBack} style={{ background: accent, color: "#fff", border: "none", borderRadius: 14, padding: "13px 32px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>← Back to Games</button>
    </div>
  );

  const card = deck[curr % deck.length];
  const pct = (timeLeft / TOTAL_TIME) * 100;

  return (
    <div style={{ maxWidth: 520, margin: "0 auto" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
        <button onClick={onBack} style={{ background: "none", border: "none", cursor: "pointer", color: C.muted, fontSize: 14, display: "flex", alignItems: "center", gap: 6 }}>
          <Icon d={I.back} size={16} color={C.muted} /> All Games
        </button>
        <span style={{ fontSize: 28, fontWeight: 800, color: timeLeft <= 10 ? C.red : accent }}>{timeLeft}s</span>
        <span style={{ fontSize: 16, fontWeight: 700, color: accent }}>✓ {score}</span>
      </div>
      <div style={{ height: 8, background: C.border, borderRadius: 4, marginBottom: 20, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`, background: timeLeft <= 10 ? C.red : accent, borderRadius: 4, transition: "width 1s linear, background .3s" }} />
      </div>
      <div style={{ background: flash === "correct" ? "#f0fdf4" : flash === "wrong" ? "#fef2f2" : "#f8f9ff", border: `2px solid ${flash === "correct" ? "#16a34a" : flash === "wrong" ? C.red : accent}33`, borderRadius: 18, padding: "28px 24px", marginBottom: 20, textAlign: "center", transition: "all .2s" }}>
        <p style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: 1, marginBottom: 10 }}>TYPE THE ANSWER</p>
        <p style={{ fontSize: 20, fontWeight: 600, color: C.text }}>{card.question}</p>
      </div>
      <div style={{ display: "flex", gap: 10 }}>
        <input ref={inputRef} value={inp} onChange={e => setInp(e.target.value)}
          onKeyDown={e => e.key === "Enter" && submit()}
          placeholder="Answer…"
          style={{ flex: 1, border: `2px solid ${C.border}`, borderRadius: 12, padding: "12px 16px", fontSize: 15, outline: "none", color: C.text }} />
        <button onClick={submit} disabled={!inp.trim()}
          style={{ background: accent, color: "#fff", border: "none", borderRadius: 12, padding: "12px 22px", fontSize: 15, fontWeight: 700, cursor: "pointer" }}>Go</button>
      </div>
      <p style={{ textAlign: "center", color: C.muted, fontSize: 12, marginTop: 10 }}>Press Enter to submit fast!</p>
    </div>
  );
}


// ─── GAME: VOICE ANSWER ───────────────────────────────────────────────────────
function VoiceAnswer({ cards, onBack }) {
  const accent = "#7c3aed";
  const [deck] = useState(() => [...cards].sort(() => Math.random() - .5));
  const [curr, setCurr] = useState(0);
  const [score, setScore] = useState(0);
  const [done, setDone] = useState(false);

  // Recording state
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState(null); // { ok, heard, correct }
  const [voiceCountdown, setVoiceCountdown] = useState(null);
  const recognitionRef = useRef(null);
  const isListeningRef = useRef(false);
  const countdownRef = useRef(null);

  const card = deck[curr];

  const startListening = () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert("Speech recognition not supported. Please use Chrome or Edge."); return; }

    // Cancel any existing countdown
    if (countdownRef.current) { clearInterval(countdownRef.current); countdownRef.current = null; }
    setVoiceCountdown(null);
    setTranscript("");
    setResult(null);
    isListeningRef.current = true;

    const rec = new SR();
    rec.continuous = true;   // keep listening until user stops
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.maxAlternatives = 1;

    let lastFinal = "";
    let silenceTimer = null;

    rec.onresult = (e) => {
      // If countdown was running and user speaks again — cancel it
      if (countdownRef.current) {
        clearInterval(countdownRef.current);
        countdownRef.current = null;
        setVoiceCountdown(null);
      }

      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) lastFinal += e.results[i][0].transcript + " ";
        else interim += e.results[i][0].transcript;
      }
      setTranscript((lastFinal + interim).trim());

      // Reset silence detection — start countdown only after 0.6s of silence
      clearTimeout(silenceTimer);
      if (lastFinal.trim()) {
        silenceTimer = setTimeout(() => {
          // Stop recognition — onend will start the countdown
          try { rec.stop(); } catch {}
        }, 600);
      }
    };

    rec.onerror = (e) => {
      if (e.error === "not-allowed") { setTranscript("Microphone access denied."); setListening(false); }
    };

    rec.onend = () => {
      clearTimeout(silenceTimer);
      isListeningRef.current = false;
      setListening(false);
      if (!lastFinal.trim()) return;
      // Wait 5 seconds, then start the 3-second countdown
      const waitTimer = setTimeout(() => {
        let countdown = 3;
        setVoiceCountdown(countdown);
        const timer = setInterval(() => {
          countdown -= 1;
          setVoiceCountdown(countdown);
          if (countdown <= 0) {
            clearInterval(timer);
            countdownRef.current = null;
            setVoiceCountdown(null);
            checkAnswer(lastFinal.trim());
          }
        }, 400);
        countdownRef.current = timer;
      }, 5000);
      // Store wait timer so it can be cancelled if user taps mic again
      countdownRef.current = waitTimer;
    };

    rec.start();
    recognitionRef.current = rec;
    setListening(true);
  };

  const stopListening = () => {
    isListeningRef.current = false;
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch {}
    }
  };

  const checkAnswer = async (heard) => {
    if (!heard.trim()) { setTranscript("Nothing heard. Try again."); return; }
    setChecking(true);
    try {
      const correct = card.answer;
      // Use AI to judge if the spoken answer matches the correct answer
      const judgment = await callClaude(
        `You are grading a spoken quiz answer. Reply with ONLY "YES" or "NO".
YES = the spoken answer is correct or close enough (synonyms, minor mispronunciation, partial but clearly correct).
NO = the spoken answer is wrong or off-topic.`,
        `Question: ${card.question}
Correct answer: ${correct}
Student said: "${heard}"
Is the student correct? Reply only YES or NO.`
      );
      const ok = judgment.trim().toUpperCase().startsWith("YES");
      setResult({ ok, heard: heard.trim(), correct });
      if (ok) setScore(s => s + 1);
    } catch(e) {
      // Fallback: simple string match
      const ok = heard.trim().toLowerCase().includes(card.answer.trim().toLowerCase()) ||
                 card.answer.trim().toLowerCase().includes(heard.trim().toLowerCase().split(" ")[0]);
      setResult({ ok, heard: heard.trim(), correct: card.answer });
      if (ok) setScore(s => s + 1);
    }
    setChecking(false);
  };

  const next = () => {
    if (curr + 1 >= deck.length) { setDone(true); return; }
    setCurr(c => c + 1);
    setTranscript("");
    setResult(null);
  };

  if (done) return <GResults score={score} total={deck.length} onBack={onBack} msg="Voice Round Done! 🎙️" />;

  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>
      <GHeader title="Voice Answer" score={score} curr={curr} total={deck.length} onBack={onBack} accent={accent} />

      {/* Question card */}
      <div style={{ background: "linear-gradient(135deg,#f5f3ff,#ede9fe)", border: `2px solid ${accent}22`, borderRadius: 20, padding: "30px 28px", marginBottom: 24, textAlign: "center" }}>
        <p style={{ fontSize: 11, fontWeight: 700, color: accent, letterSpacing: 1, marginBottom: 14, textTransform: "uppercase" }}>Speak your answer</p>
        <p style={{ fontSize: 21, fontWeight: 600, color: "#1a1a2e", lineHeight: 1.5 }}>{card.question}</p>
      </div>

      {/* Mic button */}
      {result === null && !checking && (
        <div style={{ textAlign: "center", marginBottom: 20 }}>
          <button
            onClick={() => {
              if (voiceCountdown !== null) {
                // Cancel countdown and restart listening
                clearInterval(countdownRef.current);
                countdownRef.current = null;
                setVoiceCountdown(null);
                setTranscript("");
                startListening();
              } else if (listening) {
                stopListening();
              } else {
                startListening();
              }
            }}
            style={{
              width: 88, height: 88, borderRadius: "50%", border: "none", cursor: "pointer",
              background: listening ? "#dc2626" : accent,
              boxShadow: listening ? "0 0 0 12px rgba(220,38,38,.2), 0 0 0 24px rgba(220,38,38,.08)" : `0 4px 20px ${accent}44`,
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 36,
              transition: "all .3s", margin: "0 auto",
              animation: listening ? "pulse 1.2s infinite" : "none",
            }}>
            {listening ? "⏹" : "🎙️"}
          </button>
          <p style={{ marginTop: 14, fontSize: 13, color: listening ? "#dc2626" : voiceCountdown !== null ? "#7c3aed" : "#6b7280", fontWeight: 600 }}>
            {listening ? "Listening… tap to stop" : voiceCountdown !== null ? "Tap mic to speak again" : "Tap to speak your answer"}
          </p>
          {transcript && !listening && !voiceCountdown && (
            <p style={{ marginTop: 8, fontSize: 13, color: "#6b7280", fontStyle: "italic" }}>
              Heard: "{transcript}"
            </p>
          )}
          {listening && transcript && (
            <p style={{ marginTop: 8, fontSize: 13, color: accent, fontStyle: "italic" }}>
              "{transcript}"
            </p>
          )}
          {voiceCountdown !== null && (
            <div style={{ marginTop: 16, textAlign: "center" }}>
              <p style={{ fontSize: 13, color: "#6b7280", marginBottom: 6, fontStyle: "italic" }}>Heard: "{transcript}"</p>
              <div style={{ display: "inline-flex", alignItems: "center", gap: 10, background: "#f5f3ff", border: `2px solid ${accent}33`, borderRadius: 12, padding: "10px 20px" }}>
                <div style={{ width: 36, height: 36, borderRadius: "50%", background: accent, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, fontWeight: 800 }}>{voiceCountdown}</div>
                <div>
                  <p style={{ fontSize: 13, fontWeight: 700, color: accent }}>Submitting in {voiceCountdown}… tap mic to redo</p>
                  <p style={{ fontSize: 11, color: "#6b7280" }}>Not right? tap Try again below</p>
                </div>
              </div>
              <div style={{ marginTop: 12 }}>
                <button onClick={() => { clearInterval(countdownRef.current); setVoiceCountdown(null); setTranscript(""); }}
                  style={{ background: "#f3f4f6", color: "#374151", border: "none", borderRadius: 10, padding: "8px 18px", fontSize: 13, fontWeight: 600, cursor: "pointer" }}>
                  🔄 Redo answer
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Checking */}
      {checking && (
        <div style={{ textAlign: "center", padding: "20px 0" }}>
          <div style={{ display: "flex", gap: 6, justifyContent: "center", marginBottom: 10 }}>
            {[0,1,2].map(j => <div key={j} style={{ width: 8, height: 8, borderRadius: "50%", background: accent, animation: "bounce 1.2s infinite", animationDelay: `${j * .2}s` }} />)}
          </div>
          <p style={{ fontSize: 13, color: "#6b7280" }}>Checking your answer…</p>
        </div>
      )}

      {/* Result */}
      {result !== null && (
        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 52, marginBottom: 12 }}>{result.ok ? "✅" : "❌"}</div>
          <p style={{ fontSize: 18, fontWeight: 700, color: result.ok ? "#16a34a" : "#dc2626", marginBottom: 8 }}>
            {result.ok ? "Correct!" : "Not quite"}
          </p>
          <div style={{ background: "#f9fafb", border: `1px solid ${result.ok ? "#16a34a" : "#dc2626"}33`, borderRadius: 12, padding: "14px 18px", marginBottom: 20, textAlign: "left" }}>
            <p style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>You said:</p>
            <p style={{ fontSize: 14, color: "#1a1a2e", fontStyle: "italic", marginBottom: result.ok ? 0 : 10 }}>"{result.heard}"</p>
            {!result.ok && (
              <>
                <p style={{ fontSize: 12, color: "#6b7280", marginBottom: 4 }}>Correct answer:</p>
                <p style={{ fontSize: 14, fontWeight: 700, color: "#16a34a" }}>{result.correct}</p>
              </>
            )}
          </div>
          <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
            <button onClick={() => { clearInterval(countdownRef.current); setVoiceCountdown(null); setResult(null); setTranscript(""); }}
              style={{ background: "#f3f4f6", color: "#374151", border: "none", borderRadius: 12, padding: "11px 22px", fontSize: 14, fontWeight: 600, cursor: "pointer" }}>
              🔄 Try again
            </button>
            <button onClick={next}
              style={{ background: accent, color: "#fff", border: "none", borderRadius: 12, padding: "11px 22px", fontSize: 14, fontWeight: 700, cursor: "pointer" }}>
              {curr + 1 >= deck.length ? "See Results" : "Next →"}
            </button>
          </div>
        </div>
      )}
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
