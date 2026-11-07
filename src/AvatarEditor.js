import ColorField from "./ColorPicker";
import useActionHistory from "./useActionHistory";
import { lockBodyScroll } from "./bodyScrollLock";
import { useEffect, useRef, useState } from "react";
import { ITEM_TYPES } from "./avatarItemTypes";
import ChoiceMenu from "./ChoiceMenu";
import AvatarDesigner from "./AvatarDesigner";
import AvatarGallery from "./AvatarGallery";
import { applyDesign, validDesign } from "./avatarDesigns";
import FriendlyAvatar from "./FriendlyAvatar";
import UiIcon from "./UiIcon";
import { AVATAR_COLORS, AVATAR_OPTIONS, DEFAULT_CHARACTER, randomCharacter } from "./avatarOptions";

const hex = (value, fallback) => /^#[\da-f]{6}$/i.test(value || "") ? value : /^#[\da-f]{3}$/i.test(value || "") ? "#" + [...value.slice(1)].map(x => x + x).join("") : fallback;
export default function AvatarEditor({ character, onChange, onClose, getToken }) {
  const [designing, setDesigning] = useState(null), [gallery, setGallery] = useState(false);
  const [editingDesign,setEditingDesign] = useState(null);
  const panel = useRef(null), closeRef = useRef(onClose); closeRef.current = onClose;
  const [draft,setDraft,history] = useActionHistory(() => ({...DEFAULT_CHARACTER,...character}),panel);
  const [tab,setTab] = useState("Face"), [yaw,setYaw] = useState(-.10);
  const wearDesign = design => { setDraft(current => ({ ...applyDesign(current, design), designs:(current.designs||[]).some(item=>item.id===design.id)?current.designs.map(item=>item.id===design.id?{...design,unfinished:false}:item):(current.designs||[]).length<4?[{...design,unfinished:false},...(current.designs||[])]:current.designs })); setDesigning(null); setGallery(false); setEditingDesign(null); };
  const [previewDesign,setPreviewDesign]=useState(null);
  const previewCharacter=previewDesign&&validDesign(previewDesign)?applyDesign(draft,previewDesign):draft;
  const patch = (key, value) => setDraft(current => ({ ...current, [key]: value }));
  useEffect(() => {
    const before = document.activeElement, unlock = lockBodyScroll();
    panel.current.querySelector("button")?.focus({ preventScroll: true });
    const keyboard = event => {
      if (event.key === "Escape") closeRef.current();
      if (event.key !== "Tab") return;
      const controls = [...panel.current.querySelectorAll("button:not(:disabled),input:not(:disabled),select,textarea,[tabindex]")];
      if (event.shiftKey && document.activeElement === controls[0]) { event.preventDefault(); controls.at(-1)?.focus(); }
      else if (!event.shiftKey && document.activeElement === controls.at(-1)) { event.preventDefault(); controls[0]?.focus(); }
    };
    window.addEventListener("keydown", keyboard);
    return () => { unlock(); window.removeEventListener("keydown", keyboard); if (before?.isConnected) before.focus({ preventScroll: true }); };
  }, []);
  const palette = (key, label) => <fieldset className="avatar-field"><legend>{label}</legend><div className="avatar-colors">{AVATAR_COLORS[key].map(value => <button key={value} type="button" aria-label={`${label} ${value}`} aria-pressed={draft[key]?.toLowerCase() === value} style={{ background: value }} onClick={() => patch(key, value)} />)}
    <label className="avatar-custom-color">Custom<ColorField aria-label={`Custom ${label.toLowerCase()}`} value={hex(draft[key], AVATAR_COLORS[key][0])} onChange={event => patch(key, event.target.value)} /></label></div></fieldset>;
  const customSlot={hairStyle:"customHair",topStyle:"customClothing",hat:"customHat",glasses:"customGlasses",necklace:"customNecklace",earring:"customEarring"};
  const choices = (key, label, extra = {}) => <fieldset className="avatar-field"><legend>{label}</legend><div className={`avatar-option-grid for-${key}`}>
    {AVATAR_OPTIONS[key].map((name, value) => <button type="button" key={name} aria-label={`${label}: ${name}`} aria-pressed={(draft[key] || 0) === value} onClick={() => setDraft(current=>({...current,[key]:value,...(customSlot[key]?{[customSlot[key]]:null}:{})}))}>
      <span className="avatar-option-visual"><FriendlyAvatar character={{ ...draft, ...extra, [key]: value, ...(customSlot[key]?{[customSlot[key]]:null}:{}) }} framing={key === "topStyle" ? "full" : "portrait"} size={72} /></span><span>{name}</span>{(draft[key] || 0) === value && <span className="avatar-selection-mark"><UiIcon name="check" size={12} /></span>}
    </button>)}</div></fieldset>;
  const detailColor = (key, label) => <label className="avatar-detail-color">{label}<ColorField aria-label={label} value={hex(draft[key], "#739db8")} onChange={event => patch(key, event.target.value)} /></label>;
  return <div className="avatar-editor-overlay" onMouseDown={event => { if (event.target === event.currentTarget) onClose(); }}><section className="avatar-editor premium-avatar-editor" ref={panel} role="dialog" aria-modal="true" aria-labelledby="avatar-title">
    <header><div><p className="eyebrow">MAKE IT YOURS</p><h2 id="avatar-title">Your Classio character</h2></div><button type="button" className="quiet-button" aria-label="Close character editor" onClick={onClose}><UiIcon name="close" /></button></header>
    <div className="avatar-editor-body"><div className="avatar-preview-column"><div className="avatar-stage avatar-stage-3d"><FriendlyAvatar character={previewCharacter} framing="full" yaw={yaw} size={240} /></div><div className="avatar-turn-controls"><button className="quiet-button" aria-label="Turn character left" onClick={() => setYaw(value => value - .45)}><UiIcon name="back" /></button><span>3D preview</span><button className="quiet-button" aria-label="Turn character right" onClick={() => setYaw(value => value + .45)}><UiIcon name="arrow" /></button></div>
      <label>Display name<input maxLength={40} value={draft.name || ""} onChange={event => patch("name", event.target.value)} placeholder="Your name" /></label>
      <label>Randomizer preset<ChoiceMenu aria-label="Randomizer preset" value={draft.randomizerPreset || "any"} onChange={event => patch("randomizerPreset",event.target.value)}><option value="any">Random / Any</option><option value="male">Male</option><option value="female">Female</option></ChoiceMenu></label>
      <div className="avatar-preview-actions"><button type="button" className="quiet-button" onClick={() => setDraft(current => randomCharacter(current, Math.random, current.randomizerPreset || "any"))}><UiIcon name="shuffle" />Randomize</button>
        <button type="button" className="text-button" onClick={() => setDraft({ ...DEFAULT_CHARACTER, name: draft.name || "" })}>Reset look</button></div><p>Changes are saved when you choose Save changes.</p></div>
      <div className="avatar-customization"><nav className="workspace-tabs" aria-label="Character options">{["Face", "Hair", "Outfit", "Details"].map(name => <button type="button" key={name} aria-pressed={tab === name} onClick={() => { setTab(name); setDesigning(null); setGallery(false); }}>{name}</button>)}</nav>
        <div className="avatar-editor-options">
          {designing ? <AvatarDesigner key={designing + (editingDesign?.id || "")} kind={designing} initial={editingDesign} character={draft} onPreview={setPreviewDesign} getToken={getToken} onBack={() => setDesigning(null)} onApply={wearDesign} onDraft={design => { if((draft.designs||[]).length>=4&&!draft.designs.some(item=>item.id===design.id))throw new Error("Your four saved design slots are full. Remove a saved design before adding another draft; worn items are kept.");setDraft(current=>({...current,designs:[design,...(current.designs||[]).filter(item=>item.id!==design.id)]})); }} /> : gallery ? <AvatarGallery character={draft} getToken={getToken} onClose={() => setGallery(false)} onApply={wearDesign} /> : <>
          <div className="avatar-create-actions"><button className="quiet-button" onClick={() => { setEditingDesign(null); setDesigning("clothing"); }}>Create Custom Clothing</button><button className="quiet-button" onClick={() => { setEditingDesign(null); setDesigning("hair"); }}>Create Custom Hairstyle</button>{["hat","shoes","glasses","accessory"].map(kind => <button className="quiet-button" key={kind} onClick={() => { setEditingDesign(null); setDesigning(kind); }}>Create Custom {ITEM_TYPES[kind].label}</button>)}<button className="text-button" onClick={() => setGallery(true)}>Community designs</button></div>
          {["customClothing","customHair","customHat","customShoes","customGlasses","customAccessory","customBottom","customNecklace","customEarring"].filter(key=>draft[key]).map(key=><span className="avatar-worn-item" key={key}><button className="text-button" onClick={()=>{setEditingDesign(draft[key]);setDesigning(draft[key].kind);}}>Edit {draft[key].name}</button><button className="text-button" onClick={()=>patch(key,null)}>Remove item</button></span>)}
          {!!draft.designs?.length && <div className="avatar-private-designs"><h3>Private designs & drafts</h3><p>Four saved design slots. Worn items stay available above.</p>{draft.designs.filter(validDesign).map((item,i) => <div key={item.id || i}><img src={item.texture} alt={item.name} width="40" height="40"/><span>{item.name}{item.unfinished ? " · Draft" : ""}</span><button className="quiet-button" onClick={() => wearDesign(item)}>Wear</button><button className="quiet-button" onClick={() => {setEditingDesign(item);setDesigning(item.kind);}}>Edit design</button><button className="text-button" onClick={() => setDraft(current => ({...current,designs:current.designs.filter(value=>value.id!==item.id)}))}>Remove design</button></div>)}</div>}
          {tab === "Face" && <>{palette("skin", "Skin tone")}{palette("eyes", "Eye color")}{choices("eyeShape", "Eyes", { glasses: 0 })}{choices("mouth", "Expression", { facialHair: 0 })}{choices("eyebrow", "Eyebrows")}
            <div className="avatar-toggles"><label><input type="checkbox" checked={!!draft.blush} onChange={event => patch("blush", event.target.checked)} />Rosy cheeks</label><label><input type="checkbox" checked={!!draft.freckles} onChange={event => patch("freckles", event.target.checked)} />Freckles</label></div></>}
          {tab === "Hair" && <>{choices("hairStyle", "Hairstyle", { hat: 0 })}{palette("hair", "Hair color")}</>}
          {tab === "Outfit" && <>{choices("topStyle", "Clothing")}{palette("top", "Clothing color")}{palette("bg", "Background")}</>}
          {tab === "Details" && <>{choices("hat", "Hat")}{draft.hat > 0 && detailColor("hatColor", "Hat color")}{choices("glasses", "Glasses")}{draft.glasses > 0 && detailColor("glassesColor", "Glasses color")}{choices("facialHair", "Facial hair")}{choices("necklace", "Necklace")}{draft.necklace > 0 && detailColor("necklaceColor", "Necklace color")}{choices("earring", "Earrings")}{draft.earring > 0 && detailColor("earringColor", "Earring color")}</>}
          </>}
        </div>
      </div>
    </div><footer><button type="button" className="quiet-button" disabled={!history.canUndo} onClick={history.undo}>Undo look</button><button type="button" className="quiet-button" disabled={!history.canRedo} onClick={history.redo}>Redo look</button><button type="button" className="quiet-button" onClick={onClose}>Cancel</button><button type="button" className="quiet-button is-primary" onClick={() => { onChange({ ...draft, name: draft.name.trim() }); onClose(); }}>Save changes</button></footer>
  </section></div>;
}
