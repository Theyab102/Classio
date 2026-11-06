import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import UiIcon from "./UiIcon";

export default function ActionMenu({ label, icon = "more", items, primary = false, children }) {
  const [open, setOpen] = useState(false);
  const [branch, setBranch] = useState(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const trigger = useRef(null), menu = useRef(null);
  const id = useId();
  const current = branch?.items || items;
  const close = () => { setOpen(false); setBranch(null); trigger.current?.focus({ preventScroll: true }); };

  useLayoutEffect(() => {
    if (!open || !menu.current) return;
    const positionMenu = event => {
      if (event?.target instanceof Node && menu.current?.contains(event.target)) return;
      const box = trigger.current.getBoundingClientRect();
      const height = Math.min(menu.current.scrollHeight, window.innerHeight - 24);
      setPosition({ left: Math.max(12, Math.min(box.right - 224, window.innerWidth - 236)),
        top: Math.max(12, box.bottom + height + 8 > window.innerHeight ? box.top - height - 6 : box.bottom + 6) });
    };
    positionMenu();
    menu.current.querySelector("button:not(:disabled)")?.focus({ preventScroll: true });
    window.addEventListener("resize", positionMenu);
    window.addEventListener("scroll", positionMenu, true);
    return () => { window.removeEventListener("resize", positionMenu); window.removeEventListener("scroll", positionMenu, true); };
  }, [open, branch]);

  useEffect(() => {
    if (!open) return;
    const outside = event => { if (!menu.current?.contains(event.target) && !trigger.current?.contains(event.target)) setOpen(false); };
    document.addEventListener("pointerdown", outside);
    return () => {
      document.removeEventListener("pointerdown", outside);
    };
  }, [open]);

  const keyboard = event => {
    const buttons = [...menu.current.querySelectorAll("button:not(:disabled)")];
    const index = buttons.indexOf(document.activeElement);
    if (event.key === "Escape") { event.preventDefault(); close(); }
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1
        : (index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
      buttons[next]?.focus();
    }
    if (event.key === "Tab") setOpen(false);
  };

  return <>
    <button ref={trigger} type="button" className={primary ? "quiet-button is-primary" : "quiet-button"} aria-label={label}
      aria-haspopup="menu" aria-expanded={open} aria-controls={open ? id : undefined}
      onClick={() => { setBranch(null); setOpen(value => !value); }}><UiIcon name={icon} />{children}</button>
    {open && createPortal(<div ref={menu} id={id} role="menu" aria-label={branch?.label || label} className="action-popover"
      style={position} onKeyDown={keyboard}>
      {branch && <button type="button" role="menuitem" onClick={() => setBranch(null)}><UiIcon name="back" />Back</button>}
      {current.map((item, index) => item.separator ? <hr key={index} /> : <button type="button" role="menuitem" key={item.label}
        disabled={item.disabled} className={item.danger ? "is-danger" : undefined}
        onClick={() => { if (item.items) setBranch(item); else { close(); item.run?.(); } }}>
        {item.icon && <UiIcon name={item.icon} />}<span>{item.label}</span>{item.items && <UiIcon name="chevron" />}
      </button>)}
    </div>, document.body)}
  </>;
}
