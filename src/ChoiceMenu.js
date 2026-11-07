import { Children, Fragment, isValidElement, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import UiIcon from "./UiIcon";
import "./choices.css";

function optionsFrom(children, group = "") {
  return Children.toArray(children).flatMap(child => {
    if (!isValidElement(child)) return [];
    if (child.type === Fragment) return optionsFrom(child.props.children, group);
    if (child.type === "optgroup") return optionsFrom(child.props.children, child.props.label);
    const label = Children.toArray(child.props.children).join("");
    return [{ ...child.props, label, value: String(child.props.value ?? label), group }];
  });
}

// A select-only combobox: focus stays on its trigger, including inside dialogs.
export default function ChoiceMenu({ children, value, defaultValue, onChange, disabled, className = "", style,
  variant = "", icon, descriptions = {}, onKeyDown, ...props }) {
  const options = useMemo(()=>optionsFrom(children),[children]), [local, setLocal] = useState(defaultValue ?? options[0]?.value ?? "");
  const selected = String(value ?? local), current = options.find(option => option.value === selected);
  const [open, setOpen] = useState(false), [active, setActive] = useState(0), [position, setPosition] = useState({});
  const trigger = useRef(null), menu = useRef(null), search = useRef({ text: "", at: 0 }), id = useId();
  const enabled = options.map((option, index) => !option.disabled ? index : -1).filter(index => index >= 0);
  const close = (focus = false) => { setOpen(false); if (focus) trigger.current?.focus({ preventScroll: true }); };
  const show = () => { if (!disabled && enabled.length) { const found = options.findIndex(option => option.value === selected && !option.disabled); setActive(found >= 0 ? found : enabled[0]); setOpen(true); } };
  const choose = index => {
    const option = options[index]; if (!option || option.disabled) return;
    setLocal(option.value); onChange?.({ target: { value: option.value, name: props.name }, currentTarget: { value: option.value } }); close(true);
  };
  useLayoutEffect(() => {
    if (!open) return;
    const place = event => {
      if (event?.target instanceof Node && menu.current?.contains(event.target)) return;
      const box = trigger.current.getBoundingClientRect(), width = Math.min(Math.max(box.width, variant === "notebook" ? 315 : Math.min(240,Math.max(140,...options.map(option=>option.label.length*7+50)))), window.innerWidth - 24);
      const roomBelow = window.innerHeight - box.bottom - 18, above = roomBelow < 190 && box.top > roomBelow;
      const height = Math.min(360, Math.max(80, above ? box.top - 18 : roomBelow));
      setPosition({ width, left: Math.max(12, Math.min(box.left, window.innerWidth - width - 12)), maxHeight: height,
        ...(above ? { bottom: window.innerHeight - box.top + 7 } : { top: box.bottom + 7 }) });
    };
    place(); window.addEventListener("resize", place); window.addEventListener("scroll", place, true);
    return () => { window.removeEventListener("resize", place); window.removeEventListener("scroll", place, true); };
  }, [open, variant, options]);
  useEffect(() => {
    if (!open) return;
    const outside = event => { if (!trigger.current?.contains(event.target) && !menu.current?.contains(event.target)) close(); };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  useEffect(() => { if (open) menu.current?.querySelector('[data-active="true"]')?.scrollIntoView?.({ block: "nearest" }); }, [active, open]);
  const key = event => {
    onKeyDown?.(event); if (event.defaultPrevented) return;
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault(); event.stopPropagation(); if (!open) { show(); return; }
      const i = enabled.indexOf(active);
      setActive(event.key === "Home" ? enabled[0] : event.key === "End" ? enabled.at(-1) : enabled[(i + (event.key === "ArrowDown" ? 1 : -1) + enabled.length) % enabled.length]);
    } else if (event.key === "Escape" && open) { event.preventDefault(); event.stopPropagation(); close(true); }
    else if ((event.key === "Enter" || event.key === " ") && open) { event.preventDefault(); event.stopPropagation(); choose(active); }
    else if (event.key === "Tab") close();
    else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && event.key !== " ") {
      const now = Date.now(); search.current = { text: (now - search.current.at < 700 ? search.current.text : "") + event.key.toLowerCase(), at: now };
      const found = options.findIndex(option => !option.disabled && option.label.toLowerCase().startsWith(search.current.text));
      if (found >= 0) { event.preventDefault(); if (open) setActive(found); else choose(found); }
    }
  };
  return <>
    <button {...props} ref={trigger} type="button" disabled={disabled} value={selected} role="combobox" aria-haspopup="listbox"
      aria-expanded={open} aria-controls={open ? id : undefined} aria-activedescendant={open ? id + "-" + active : undefined}
      className={"choice-trigger " + className + (variant ? " choice-" + variant : "")} style={style} onKeyDown={key}
      onClick={() => open ? close() : show()}>
      {icon && <UiIcon name={icon} size={18} />}<span className="choice-value">{variant === "notebook" && <small>NOTEBOOK MODE</small>}{current?.label || props.placeholder || "Choose an option"}</span><UiIcon className="choice-chevron" name="down" size={17} />
    </button>
    {open && createPortal(<div ref={menu} id={id} role="listbox" aria-label={props["aria-label"] || "Options"} className="choice-popover" style={position}>
      {options.map((option, index) => <Fragment key={option.value}>
        {option.group && option.group !== options[index - 1]?.group && <p className="choice-group">{option.group}</p>}
        <button type="button" id={id + "-" + index} role="option" data-value={option.value} tabIndex={-1} disabled={option.disabled} aria-selected={option.value === selected}
          data-active={active === index} onMouseDown={event => event.preventDefault()} onPointerMove={() => setActive(index)} onClick={() => choose(index)}>
          <span><strong>{option.label}</strong>{descriptions[option.value] && <small>{descriptions[option.value]}</small>}</span>{option.value === selected && <UiIcon name="check" size={17} />}
        </button>
      </Fragment>)}
    </div>, document.body)}
  </>;
}
