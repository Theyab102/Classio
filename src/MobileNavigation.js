import { useState } from 'react';
import Dialog from './Dialog';
import { useLocale } from './locale';

export default function MobileNavigation({ navigation, contextTabs, onNavigate, onTab, activeTab, onSearch, onProfile, onTheme, onSignOut }) {
  const { t } = useLocale();
  const [open, setOpen] = useState(false);
  function select(action) { action(); setOpen(false); }
  return <>
    <header className="mobile-header"><button type="button" aria-label={t('Open navigation menu')} aria-expanded={open} onClick={() => setOpen(true)}><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 6h16M4 12h16M4 18h16" /></svg></button><a href="/" onClick={event => { event.preventDefault(); onNavigate('home'); }}>Classio</a><button type="button" onClick={onSearch} aria-label={t('Search your workspace')}>⌕</button></header>
    {open && <Dialog onClose={() => setOpen(false)} label={t('Navigation menu')} className="navigation-drawer"><h2>{t('Your workspace')}</h2>
      <nav aria-label={t('Main navigation')}>{navigation.map(item => <button key={item.id} type="button" onClick={() => select(() => onNavigate(item.id))}>{item.icon}<span>{item.label}</span></button>)}</nav>
      {!!contextTabs?.length && <nav aria-label={t('Current study tools')}><h3>{t('Study tools')}</h3>{contextTabs.map(item => <button key={item.id} type="button" aria-current={activeTab === item.id ? 'page' : undefined} onClick={() => select(() => onTab(item.id))}>{item.icon}<span>{item.label}</span></button>)}</nav>}
      <div className="drawer-account"><button type="button" onClick={() => select(onProfile)}>{t('Your avatar')}</button><button type="button" onClick={onTheme}>{t('Switch appearance')}</button><button type="button" onClick={() => select(onSignOut)}>{t('Sign out / exit guest session')}</button></div>
    </Dialog>}
  </>;
}
