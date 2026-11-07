import ChoiceMenu from "./ChoiceMenu";
import { useEffect, useSyncExternalStore } from "react";

const key = "classio_theme";
const modes = new Set(["system", "light", "dark"]);
const listeners = new Set();
let media;

function readPreference() {
  try { return modes.has(localStorage.getItem(key)) ? localStorage.getItem(key) : "system"; }
  catch { return "system"; }
}

let preference = readPreference();
const deviceIsDark = () => Boolean(window.matchMedia?.("(prefers-color-scheme: dark)").matches);
export const activeTheme = () => preference === "system" ? (deviceIsDark() ? "dark" : "light") : preference;
const snapshot = () => `${preference}:${activeTheme()}`;

function notify() {
  for (const listener of listeners) listener();
}

function storageChanged(event) {
  if (event.key === key || event.key === null) { preference = readPreference(); notify(); }
}

function subscribe(listener) {
  listeners.add(listener);
  if (listeners.size === 1) {
    preference = readPreference();
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
  preference = value;
  try { localStorage.setItem(key, value); } catch {}
  notify();
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
