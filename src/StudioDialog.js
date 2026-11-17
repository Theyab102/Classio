import useVisualViewport from "./useVisualViewport";
import { lockBodyScroll } from "./bodyScrollLock";
import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";
import UiIcon from "./UiIcon";

export default function StudioDialog({ title, onClose, children, wide = false }) {
  const viewport=useRef(null);useVisualViewport(viewport);
  const ref = useRef(null), close = useRef(onClose), id = useId(); close.current = onClose;
  useEffect(() => {
    const before = document.activeElement, unlock = lockBodyScroll();

    const content=ref.current.querySelector('.studio-dialog-content');
    (content.querySelector('input:not([type=checkbox]):not([type=hidden]):not([type=file]):not([hidden]):not([disabled]),textarea:not([hidden]):not([disabled]),select:not([hidden]):not([disabled])')||content.querySelector('button:not([hidden]):not([disabled]),a[href]')||ref.current.querySelector('button'))?.focus({preventScroll:true});
    const keys = event => {
      const dialogs=document.querySelectorAll('.studio-dialog');
      if(dialogs[dialogs.length-1]!==ref.current)return;
      if (event.key === "Escape" && ref.current.querySelector(".is-presenting")) return;
      if (event.key === "Escape") { event.preventDefault(); close.current(); }
      if (event.key !== "Tab") return;
      const controls = [...ref.current.querySelectorAll("button:not(:disabled),input:not([disabled]):not([hidden]),textarea:not([disabled]),select:not([disabled]),a[href],[tabindex='0']")].filter(node=>node.getClientRects().length);
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    };
    window.addEventListener("keydown", keys);
    return () => { unlock(); window.removeEventListener("keydown", keys); if (before?.isConnected) before.focus({ preventScroll: true }); };
  }, []);
  return createPortal(<div ref={viewport} className="studio-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><section className={`studio-dialog ${wide ? "is-wide" : ""}`} ref={ref} role="dialog" aria-modal="true" aria-labelledby={id}>
    <header><h2 id={id}>{title}</h2><button className="quiet-button icon-dismiss" style={{minWidth:44,minHeight:44}} aria-label={`Close ${title}`} onClick={onClose}><UiIcon name="close" /></button></header><div className="studio-dialog-content">{children}</div></section></div>, document.fullscreenElement || document.body);
}
