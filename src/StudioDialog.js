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

    ref.current.querySelector("input:not([type=checkbox]),textarea,button")?.focus({ preventScroll: true });
    const keys = event => {
      if (event.key === "Escape") { event.preventDefault(); close.current(); }
      if (event.key !== "Tab") return;
      const controls = [...ref.current.querySelectorAll("button:not(:disabled),input,textarea,select,a[href]")];
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    };
    window.addEventListener("keydown", keys);
    return () => { unlock(); window.removeEventListener("keydown", keys); if (before?.isConnected) before.focus({ preventScroll: true }); };
  }, []);
  return createPortal(<div ref={viewport} className="studio-dialog-backdrop" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><section className={`studio-dialog ${wide ? "is-wide" : ""}`} ref={ref} role="dialog" aria-modal="true" aria-labelledby={id}>
    <header><h2 id={id}>{title}</h2><button className="quiet-button" aria-label={`Close ${title}`} onClick={onClose}><UiIcon name="close" /></button></header><div className="studio-dialog-content">{children}</div></section></div>, document.body);
}
