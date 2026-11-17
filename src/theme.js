import ChoiceMenu from "./ChoiceMenu";
import { useEffect, useSyncExternalStore } from "react";
import {flushSync} from 'react-dom';

const key = "classio_theme";
const modes = new Set(["system", "light", "dark"]);
const listeners = new Set();
let media;

function readPreference() {
  try { return modes.has(localStorage.getItem(key)) ? localStorage.getItem(key) : "system"; }
  catch { return "system"; }
}

let preference = readPreference();
let requestedPreference=preference,transition;
const deviceIsDark = () => Boolean(window.matchMedia?.("(prefers-color-scheme: dark)").matches);
export const activeTheme = () => preference === "system" ? (deviceIsDark() ? "dark" : "light") : preference;
const snapshot = () => `${preference}:${activeTheme()}`;

function notify() {
  for (const listener of listeners) listener();
}

function storageChanged(event) {
  if (event.key === key || event.key === null) { transition?.skipTransition();requestedPreference=preference=readPreference();notify(); }
}

function subscribe(listener) {
  listeners.add(listener);
  if (listeners.size === 1) {
    requestedPreference=preference=readPreference();
    media = window.matchMedia?.("(prefers-color-scheme: dark)");
    media?.addEventListener?.("change", notify);
    window.addEventListener("storage", storageChanged);
  }
  return () => {
    listeners.delete(listener);
    if (!listeners.size) {
      media?.removeEventListener?.("change", notify);
      window.removeEventListener("storage", storageChanged);
    }
  };
}

export function setThemePreference(value) {
  if (!modes.has(value)) return;
  requestedPreference=value;
  transition?.skipTransition();
  const commit=()=>{preference=requestedPreference;try{localStorage.setItem(key,preference);}catch{}flushSync(notify);};
  const reduced=window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  if(document.startViewTransition&&!reduced){
    try{
      const next=document.startViewTransition(commit);transition=next;
      next.ready?.catch(()=>{});
      next.finished?.finally(()=>{if(transition===next)transition=null;}).catch(()=>{});
      return;
    }catch{}
  }
  document.body.classList.remove('theme-changing');commit();
  if(!reduced){void document.body.offsetWidth;document.body.classList.add('theme-changing');}
}

export function toggleThemePreference(){
  const resolved=requestedPreference==='system'?(deviceIsDark()?'dark':'light'):requestedPreference;
  setThemePreference(resolved==='dark'?'light':'dark');
}

export function useThemeMode() {
  const value = useSyncExternalStore(subscribe, snapshot, () => "system:light");
  const [mode, resolved] = value.split(":");
  useEffect(() => {
    document.documentElement.dataset.theme = resolved;
    document.documentElement.style.colorScheme = resolved;
    document.body.classList.toggle("classio-dark", resolved === "dark");
    // Remove the legacy inline background so both entry and workspace use CSS.
    document.body.style.removeProperty("background");
  }, [resolved]);
  return { mode, resolved, setMode: setThemePreference };
}

export function ThemeSelector() {
  const { mode, setMode } = useThemeMode();
  return <label className="theme-selector"><span className="theme-label">Appearance</span>
    <ChoiceMenu aria-label="Appearance" value={mode} onChange={event => setMode(event.target.value)}>
      <option value="system">Use device setting</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </ChoiceMenu>
  </label>;
}
