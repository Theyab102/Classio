import { useEffect, useRef, useState } from "react";
import FriendlyAvatar from "./FriendlyAvatar";
import UiIcon from "./UiIcon";

const hairNames = ["Close crop", "Side part", "Fringe", "Textured", "Long", "Wavy", "Ponytail", "Bun", "Afro", "Braids", "Curly", "Locs", "Faux hawk", "Two buns", "Mohawk", "Dreadlocks", "Swept back"];
const colors = {
  skin: ["#f4d6c8", "#fddbb4", "#e8a87c", "#c68642", "#a0693a", "#8d5524", "#6b3a1f", "#49291c"],
  hair: ["#242424", "#493326", "#815533", "#b8783e", "#d6b774", "#eee0bb", "#ad543e", "#777f86"],
  top: ["#417f99", "#264658", "#738e72", "#b66957", "#b99d61", "#ededdf", "#787e8b", "#a46378"],
  eyes: ["#373b38", "#775b3b", "#4f7766", "#487b9b"],
  bg: ["#e5f0f7", "#f4ead8", "#e3eddc", "#f4e1dc", "#e7e9ed", "#273e4b"],
};
const hex = (value, fallback) => /^#[\da-f]{6}$/i.test(value || "") ? value : /^#[\da-f]{3}$/i.test(value || "") ? "#" + [...value.slice(1)].map(x => x + x).join("") : fallback;

export default function AvatarEditor({ character, onChange, onClose }) {
  const [tab, setTab] = useState("Face");
  const panel = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const patch = (key, value) => onChange({ ...character, [key]: value });
  useEffect(() => {
    const before = document.activeElement, overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    panel.current.querySelector("button")?.focus();
    const keyboard = event => {
      if (event.key === "Escape") closeRef.current();
      if (event.key !== "Tab") return;
      const controls = [...panel.current.querySelectorAll("button, input, select")];
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    };
    window.addEventListener("keydown", keyboard);
    return () => { document.body.style.overflow = overflow; window.removeEventListener("keydown", keyboard); if (before?.isConnected) before.focus(); };
  }, []);

  const palette = (key, label) => <fieldset className="avatar-field"><legend>{label}</legend><div className="avatar-colors">
    {colors[key].map(value => <button key={value} type="button" aria-label={`${label} ${value}`} aria-pressed={character[key]?.toLowerCase() === value}
      style={{ background: value }} onClick={() => patch(key, value)} />)}
    <label className="avatar-custom-color">Custom<input type="color" aria-label={`Custom ${label.toLowerCase()}`} value={hex(character[key], colors[key][0])} onChange={event => patch(key, event.target.value)} /></label>
  </div></fieldset>;
  const select = (key, label, values) => <label className="avatar-select">{label}<select value={character[key] || 0} onChange={event => patch(key, Number(event.target.value))}>
    {values.map((value, index) => <option key={value} value={index}>{value}</option>)}
  </select></label>;

  return <div className="avatar-editor-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="avatar-editor" ref={panel} role="dialog" aria-modal="true" aria-labelledby="avatar-title">
      <header><div><h2 id="avatar-title">Your character</h2><p>Choose a look that feels like you.</p></div>
        <button type="button" className="quiet-button" aria-label="Close character editor" onClick={onClose}><UiIcon name="close" /></button></header>
      <div className="avatar-preview"><FriendlyAvatar character={character} size={136} />
        <label>Display name<input maxLength="40" value={character.name || ""} onChange={event => patch("name", event.target.value)} placeholder="Optional" /></label>
      </div>
      <nav className="workspace-tabs" aria-label="Character options">{["Face", "Hair", "Outfit", "Details"].map(name =>
        <button type="button" key={name} aria-pressed={tab === name} onClick={() => setTab(name)}>{name}</button>)}</nav>
      <div className="avatar-editor-options">
        {tab === "Face" && <>{palette("skin", "Skin tone")}{palette("eyes", "Eye color")}
          {select("eyeShape", "Eyes", ["Round", "Soft", "Lashes"])}{select("mouth", "Expression", ["Smile", "Frown", "Surprised", "Smirk", "Neutral"])}
          {select("eyebrow", "Eyebrows", ["Arched", "Straight", "Full", "Soft angle"])}</>}
        {tab === "Hair" && <><fieldset className="avatar-field"><legend>Hairstyle</legend><div className="avatar-hairstyles">
          {hairNames.map((name, index) => <button key={name} type="button" aria-label={name} aria-pressed={(character.hairStyle || 0) === index} onClick={() => patch("hairStyle", index)}>
            <FriendlyAvatar character={{ ...character, hairStyle: index, hat: 0 }} size={52} /><span>{name}</span></button>)}
        </div></fieldset>{palette("hair", "Hair color")}</>}
        {tab === "Outfit" && <>{select("topStyle", "Clothing", ["T-shirt", "Hoodie", "Jacket", "Tank top", "Suit", "Crop top", "Polo", "Jersey", "Striped top", "Dress", "Blazer", "Graphic tee", "Varsity", "Wrap top", "Turtleneck", "Denim"])}
          {palette("top", "Clothing color")}{palette("bg", "Background")}</>}
        {tab === "Details" && <>{select("hat", "Hat", ["None", "Beanie", "Cap", "Cowboy", "Graduation", "Crown", "Party hat", "Headband", "Bucket hat"])}
          {select("glasses", "Glasses", ["None", "Round", "Rectangle", "Cat-eye", "Sunglasses", "Hearts", "Aviator", "Monocle"])}
          {select("facialHair", "Facial hair", ["None", "Stubble", "Moustache", "Short beard", "Full beard", "Goatee"])}
          {select("necklace", "Necklace", ["None", "Chain", "Pendant", "Beads", "Choker", "Pearl"])}
          {select("earring", "Earrings", ["None", "Studs", "Drops", "Hoops", "Stars", "Dangles", "Crosses"])}
          <div className="avatar-select"><label><input type="checkbox" checked={!!character.blush} onChange={event => patch("blush", event.target.checked)} /> Rosy cheeks</label>
            <label><input type="checkbox" checked={!!character.freckles} onChange={event => patch("freckles", event.target.checked)} /> Freckles</label></div>
        </>}
      </div>
      <footer><button type="button" className="quiet-button is-primary" onClick={onClose}>Done</button></footer>
    </section>
  </div>;
}
