import { useEffect, useRef, useState } from "react";
import FriendlyAvatar from "./FriendlyAvatar";
import UiIcon from "./UiIcon";
import { AVATAR_COLORS, AVATAR_OPTIONS, DEFAULT_CHARACTER, randomCharacter } from "./avatarOptions";

const hex = (value, fallback) => /^#[\da-f]{6}$/i.test(value || "") ? value : /^#[\da-f]{3}$/i.test(value || "") ? "#" + [...value.slice(1)].map(x => x + x).join("") : fallback;
export default function AvatarEditor({ character, onChange, onClose }) {
  const [tab, setTab] = useState("Face"), [draft, setDraft] = useState(() => ({ ...DEFAULT_CHARACTER, ...character }));
  const panel = useRef(null), closeRef = useRef(onClose); closeRef.current = onClose;
  const patch = (key, value) => setDraft(current => ({ ...current, [key]: value }));
  useEffect(() => {
    const before = document.activeElement, overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden"; panel.current.querySelector("button")?.focus({ preventScroll: true });
    const keyboard = event => {
      if (event.key === "Escape") closeRef.current();
      if (event.key !== "Tab") return;
      const controls = [...panel.current.querySelectorAll("button:not(:disabled),input,select")];
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    };
    window.addEventListener("keydown", keyboard);
    return () => { document.body.style.overflow = overflow; window.removeEventListener("keydown", keyboard); if (before?.isConnected) before.focus({ preventScroll: true }); };
  }, []);
  const palette = (key, label) => <fieldset className="avatar-field"><legend>{label}</legend><div className="avatar-colors">{AVATAR_COLORS[key].map(value => <button key={value} type="button" aria-label={`${label} ${value}`} aria-pressed={draft[key]?.toLowerCase() === value} style={{ background: value }} onClick={() => patch(key, value)} />)}
    <label className="avatar-custom-color">Custom<input type="color" aria-label={`Custom ${label.toLowerCase()}`} value={hex(draft[key], AVATAR_COLORS[key][0])} onChange={event => patch(key, event.target.value)} /></label></div></fieldset>;
  const choices = (key, label, extra = {}) => <fieldset className="avatar-field"><legend>{label}</legend><div className={`avatar-option-grid for-${key}`}>
    {AVATAR_OPTIONS[key].map((name, value) => <button type="button" key={name} aria-label={`${label}: ${name}`} aria-pressed={(draft[key] || 0) === value} onClick={() => patch(key, value)}>
      <span className="avatar-option-visual"><FriendlyAvatar character={{ ...draft, ...extra, [key]: value }} size={72} /></span><span>{name}</span>{(draft[key] || 0) === value && <span className="avatar-selection-mark"><UiIcon name="check" size={12} /></span>}
    </button>)}</div></fieldset>;
  const detailColor = (key, label) => <label className="avatar-detail-color">{label}<input type="color" aria-label={label} value={hex(draft[key], "#739db8")} onChange={event => patch(key, event.target.value)} /></label>;
  return <div className="avatar-editor-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><section className="avatar-editor premium-avatar-editor" ref={panel} role="dialog" aria-modal="true" aria-labelledby="avatar-title">
    <header><div><p className="eyebrow">MAKE IT YOURS</p><h2 id="avatar-title">Your Classio character</h2></div><button type="button" className="quiet-button" aria-label="Close character editor" onClick={onClose}><UiIcon name="close" /></button></header>
    <div className="avatar-editor-body"><div className="avatar-preview-column"><div className="avatar-stage"><FriendlyAvatar character={draft} size={184} /><span className="avatar-stage-shadow" /></div>
      <label>Display name<input maxLength={40} value={draft.name || ""} onChange={event => patch("name", event.target.value)} placeholder="Your name" /></label>
      <div className="avatar-preview-actions"><button type="button" className="quiet-button" onClick={() => setDraft(current => randomCharacter(current))}><UiIcon name="shuffle" />Randomize</button>
        <button type="button" className="text-button" onClick={() => setDraft({ ...DEFAULT_CHARACTER, name: draft.name || "" })}>Reset look</button></div><p>Changes are saved when you choose Save changes.</p></div>
      <div className="avatar-customization"><nav className="workspace-tabs" aria-label="Character options">{["Face", "Hair", "Outfit", "Details"].map(name => <button type="button" key={name} aria-pressed={tab === name} onClick={() => setTab(name)}>{name}</button>)}</nav>
        <div className="avatar-editor-options">
          {tab === "Face" && <>{palette("skin", "Skin tone")}{palette("eyes", "Eye color")}{choices("eyeShape", "Eyes", { glasses: 0 })}{choices("mouth", "Expression", { facialHair: 0 })}{choices("eyebrow", "Eyebrows")}
            <div className="avatar-toggles"><label><input type="checkbox" checked={!!draft.blush} onChange={event => patch("blush", event.target.checked)} />Rosy cheeks</label><label><input type="checkbox" checked={!!draft.freckles} onChange={event => patch("freckles", event.target.checked)} />Freckles</label></div></>}
          {tab === "Hair" && <>{choices("hairStyle", "Hairstyle", { hat: 0 })}{palette("hair", "Hair color")}</>}
          {tab === "Outfit" && <>{choices("topStyle", "Clothing")}{palette("top", "Clothing color")}{palette("bg", "Background")}</>}
          {tab === "Details" && <>{choices("hat", "Hat")}{draft.hat > 0 && detailColor("hatColor", "Hat color")}{choices("glasses", "Glasses")}{draft.glasses > 0 && detailColor("glassesColor", "Glasses color")}{choices("facialHair", "Facial hair")}{choices("necklace", "Necklace")}{draft.necklace > 0 && detailColor("necklaceColor", "Necklace color")}{choices("earring", "Earrings")}{draft.earring > 0 && detailColor("earringColor", "Earring color")}</>}
        </div>
      </div>
    </div><footer><button type="button" className="quiet-button" onClick={onClose}>Cancel</button><button type="button" className="quiet-button is-primary" onClick={() => { onChange({ ...draft, name: draft.name.trim() }); onClose(); }}>Save changes</button></footer>
  </section></div>;
}
