import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useLocale } from './locale';

export default function Dialog({ children, onClose, label = 'Classio dialog', className = '' }) {
  const { t } = useLocale();
  const native=typeof HTMLDialogElement!=='undefined'&&typeof HTMLDialogElement.prototype.showModal==='function';
  const element = useRef(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const dialog = element.current;
    dialog.showModal?.();
    if (!dialog.open) dialog.setAttribute('open', '');
    const keydown=event=>{
      if(native)return;
      if(event.key==='Escape'){event.preventDefault();close.current();}
      if(event.key==='Tab'){
        const focusable=[...dialog.querySelectorAll('button,input,textarea,select,a[href],[tabindex="0"]')].filter(item=>!item.disabled&&item.getClientRects().length);
        const first=focusable[0],last=focusable.at(-1);
        if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}
        else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
      }
    };
    if(!native){dialog.querySelector('button,input,textarea,select')?.focus();window.addEventListener('keydown',keydown);}

    return () => {
      window.removeEventListener('keydown',keydown);
      dialog.close?.();
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus?.();
    };
  }, [native]);
  const content=<dialog ref={element} aria-modal="true" className={`classio-dialog ${className}`} aria-label={label}
    onCancel={event => { event.preventDefault(); close.current(); }}
    onClick={event => { if (event.target === element.current) { const rect = element.current.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) close.current(); } }}>
    <button className="dialog-close" type="button" onClick={onClose} aria-label={t("Close dialog")}>×</button>
    {children}
  </dialog>;
  return createPortal(native?content:<div className="dialog-fallback-backdrop" style={className.includes('navigation-drawer')?{display:'block'}:undefined} onClick={event=>{if(event.target===event.currentTarget)onClose();}}>{content}</div>,document.body);
}
