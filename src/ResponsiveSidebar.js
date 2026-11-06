import { useEffect, useRef, useState } from "react";
import { ThemeSelector } from "./theme";
import UiIcon from "./UiIcon";
import "./responsive.css";

function MenuIcon({ close = false }) {
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
    <path d={close ? "M6 6l12 12M6 18L18 6" : "M4 6h16M4 12h16M4 18h16"} />
  </svg>;
}

export default function ResponsiveSidebar({ mobile, expanded, onExpand, main, mainActive, tabs, active, onTab, onNavigate,
  onSearch, onCharacter, avatar, account, guest, onSignOut, onToggleTheme, dark, documentName }) {
  const [open, setOpen] = useState(false);
  const drawer = useRef(null);
  const trigger = useRef(null);
  const full = mobile || expanded;

  useEffect(() => {
    if (!mobile || !open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const previousFocus = document.activeElement;
    drawer.current?.querySelector("button")?.focus();
    const keyboard = event => {
      if (event.key === "Escape") { event.preventDefault(); setOpen(false); }
      if (event.key !== "Tab") return;
      const elements = [...drawer.current.querySelectorAll("button:not(:disabled), select, a[href]")];
      const first = elements[0], last = elements.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    window.addEventListener("keydown", keyboard);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", keyboard);
      if (trigger.current?.isConnected) trigger.current.focus();
      else if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [mobile, open]);

  const choose = callback => { callback(); setOpen(false); };
  const menu = <aside ref={drawer} id="classio-navigation" className={`classio-navigation ${full ? "is-expanded" : "is-compact"}`}
    role={mobile ? "dialog" : undefined} aria-modal={mobile ? "true" : undefined} aria-label="Workspace navigation">
    <div className="navigation-brand">
      <button type="button" onClick={() => mobile ? setOpen(false) : onExpand(value => !value)}
        aria-label={mobile ? "Close menu" : expanded ? "Collapse sidebar" : "Expand sidebar"}>
        {mobile ? <MenuIcon close /> : <img src="/ClassioLogo.png" alt="" width="28" height="28" />}
      </button>
      {full && <span>Classio</span>}
    </div>
    <div className="navigation-scroll">
      <button className="navigation-item" type="button" onClick={() => choose(onSearch)} aria-label="Search files" title="Search files">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="10" cy="10" r="7" /><path d="m15 15 6 6" /></svg>
        {full && <span>Search files</span>}
      </button>
      <nav aria-label="Main menu">
        {main.map(item => <button type="button" key={item.id} className="navigation-item" aria-label={item.label} title={item.label}
          aria-current={mainActive === item.id ? "page" : undefined}
          onClick={() => choose(() => onNavigate(item.id))}>
          {item.icon}{full && <span>{item.label}</span>}
        </button>)}
      </nav>
      {tabs.length > 0 && <nav className="navigation-tools" aria-label="Study tools">
        {full && <p className="navigation-context" title={documentName}>{documentName || "Study tools"}</p>}
        {tabs.map(item => <button type="button" key={item.id} className="navigation-item" aria-label={item.label} title={item.label}
          aria-current={active === item.id ? "page" : undefined} onClick={() => choose(() => onTab?.(item.id))}>
          {item.icon}{full && <span>{item.label}</span>}
        </button>)}
      </nav>}
      <div className="navigation-account">
        {full ? <ThemeSelector /> : <button type="button" className="navigation-item" onClick={onToggleTheme}
          aria-label={dark ? "Switch to light mode" : "Switch to dark mode"} title="Change appearance">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1 1M18 18l1 1M5 19l1-1M18 6l1-1" /></svg>
        </button>}
        <button type="button" className="navigation-item" onClick={() => choose(onCharacter)} title="Edit character" aria-label="Edit character">
          {avatar}{full && <span>{account || "Guest"}</span>}
        </button>
        {full && <p className="navigation-session">{guest ? "Guest work clears when this session ends." : "Signed in. Progress syncs with your account."}</p>}
        {full && <button type="button" className="navigation-item" onClick={() => choose(onSignOut)}>{guest ? "End guest session" : "Sign out"}</button>}
      </div>
    </div>
  </aside>;

  return <>
    <header className="workspace-topbar" style={{ insetInlineStart: mobile ? 0 : expanded ? 220 : 60 }}>
      {mobile && <button ref={trigger} type="button" className="quiet-button" aria-label="Open menu" aria-expanded={open} aria-controls="classio-navigation" onClick={() => setOpen(true)}><MenuIcon /></button>}
      {documentName && <button type="button" className="quiet-button" aria-label="Back to library" onClick={() => onNavigate("home")}><UiIcon name="back" /></button>}
      <span className="workspace-topbar-title">{documentName || "Classio"}</span>
      <button type="button" className="quiet-button" aria-label="AI Assistant" onClick={() => onNavigate("assistant")}>Ask AI</button>
    </header>
    {mobile ? open && <div className="phone-navigation-overlay">
      <button type="button" tabIndex={-1} className="navigation-backdrop" aria-label="Close menu backdrop" onClick={() => setOpen(false)} />
      {menu}
    </div> : menu}
  </>;
}
