import { useSyncExternalStore } from 'react';

export const THEMES = {
  light: { bg: '#f8fafc', surface: '#ffffff', border: '#dbe4ef', text: '#17243b', muted: '#52627a', accent: '#1d4ed8', accentL: '#eff6ff', accentS: '#bfdbfe', warm: '#996122', warmL: '#fff6e5', green: '#27734b', greenL: '#eaf7ef', purple: '#1d4ed8', purpleL: '#eff6ff', red: '#b42335', redL: '#fff0f1', sidebar: '#ffffff', sidebarActive: '#eaf2ff', navText: '#17243b', shadow: '0 2px 16px #1d4ed809', cardShadow: '0 8px 28px #1e3a5f0a', grad: '#1d4ed8', isDark: false },
  dark: { bg: '#0f172a', surface: '#1e293b', border: '#3b4a63', text: '#f1f5f9', muted: '#b8c6da', accent: '#93c5fd', accentL: '#172e50', accentS: '#395d8a', warm: '#e6b873', warmL: '#352b1e', green: '#86d9ab', greenL: '#153a2b', purple: '#93c5fd', purpleL: '#172e50', red: '#fda4af', redL: '#431e2a', sidebar: '#152035', sidebarActive: '#233753', navText: '#f1f5f9', shadow: '0 2px 16px #0003', cardShadow: '0 8px 28px #0003', grad: '#1d4ed8', isDark: true },
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
