import { useSyncExternalStore } from 'react';

export const THEMES = {
  light: { bg:'#f2f2f2', surface:'#ffffff', border:'#d3dfe3', text:'#183a44', muted:'#526b74', accent:'#2b6777', accentL:'#e7eff3', accentS:'#c8d8e4', warm:'#885b2b', warmL:'#faf1e4', green:'#276c5c', greenL:'#e5f3ee', purple:'#2b6777', purpleL:'#e7eff3', red:'#b42335', redL:'#fff0f1', sidebar:'#ffffff', sidebarActive:'#e2edf1', navText:'#183a44', shadow:'0 2px 16px #2b677708', cardShadow:'0 6px 24px #2b67770c', grad:'#2b6777', isDark:false },
  dark: { bg:'#11282f', surface:'#1b373f', border:'#49656d', text:'#f4f8f9', muted:'#bbced4', accent:'#a6d6d1', accentL:'#25464f', accentS:'#547680', warm:'#e2bd85', warmL:'#393124', green:'#8bd4b9', greenL:'#1e4439', purple:'#a6d6d1', purpleL:'#25464f', red:'#fda4af', redL:'#431e2a', sidebar:'#152f37', sidebarActive:'#284953', navText:'#f4f8f9', shadow:'0 2px 16px #0002', cardShadow:'0 6px 24px #0003', grad:'#2b6777', isDark:true },
};
let dark = false;
try { const saved = localStorage.getItem('classio_dark'); dark = saved === null ? window.matchMedia?.('(prefers-color-scheme: dark)').matches : saved === 'true'; } catch {}
const listeners = new Set();
export function getTheme() { return dark ? THEMES.dark : THEMES.light; }
function apply() {
  const theme = getTheme();
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.body.classList.toggle('classio-dark', dark);
  document.body.style.background = theme.bg;
  for (const [key, value] of Object.entries(theme)) if (typeof value === 'string') document.documentElement.style.setProperty('--theme-' + key, value);
  listeners.forEach(listener => listener());
}
export function setTheme(value) {
  dark = value === 'dark';
  try { localStorage.setItem('classio_dark', String(dark)); } catch {}
  apply();
}
export function toggleTheme() { setTheme(dark ? 'light' : 'dark'); }
function subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); }
export function useTheme() { return useSyncExternalStore(subscribe, getTheme); }
if (typeof document !== 'undefined') apply();
window.addEventListener('storage', event => { if (event.key === 'classio_dark') { dark = event.newValue === 'true'; apply(); } });
