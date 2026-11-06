import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useLocale } from './locale';

export default function Dialog({ children, onClose, label = 'Classio dialog', className = '' }) {
  const { t } = useLocale();
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
    return () => {
      dialog.close?.();
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus?.();
    };
  }, []);
  return createPortal(<dialog ref={element} className={`classio-dialog ${className}`} aria-label={label}
    onCancel={event => { event.preventDefault(); close.current(); }}
    onClick={event => { if (event.target === element.current) { const rect = element.current.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) close.current(); } }}>
    <button className="dialog-close" type="button" onClick={onClose} aria-label={t("Close dialog")}>×</button>
    {children}
  </dialog>, document.body);
}
