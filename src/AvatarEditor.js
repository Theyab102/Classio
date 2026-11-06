import { useId, useState } from 'react';
import Avatar, { avatarDefaults } from './Avatar';
import Dialog from './Dialog';
import { useLocale } from './locale';
import './avatar.css';

const choices = {
  faceShape:['Round','Oval','Soft square','Heart','Long','Wide'], nose:['Button','Soft bridge','Straight','Broad'],
  hairStyle:['Buzz','Side-part','Fringe','Textured','Long','Wavy','Ponytail','Bun','Afro','Box braids','Curly','Locs','Faux hawk','Space buns','Mohawk','Dreadlocks','Slick back'],
  eyeShape:['Round','Almond','Cat-eye','Relaxed','Bright'], eyebrow:['Arched','Straight','Thick','Sad','Raised','Gentle'],
  mouth:['Smile','Frown','Open','Smirk','Neutral','Grin','Gentle'], facialHair:['None','Stubble','Moustache','Short beard','Full beard','Goatee'],
  topStyle:['T-Shirt','Hoodie','Jacket','Tank Top','Suit & Tie','Crop Top','Polo','Jersey','Streetwear','Dress','Blazer','Graphic Tee','Varsity','Kimono','Turtleneck','Denim'],
  hat:['None','Beanie','Cap','Cowboy','Grad cap','Crown','Party hat','Headband','Bucket hat'],
  glasses:['None','Round','Rectangular','Cat-eye','Sunglasses','Heart','Aviator','Monocle'],
  earring:['None','Studs','Drops','Hoops','Stars','Dangles','Crosses'], necklace:['None','Gold chain','Pendant','Beads','Choker','Pearl'],
};
const colours = {
  skin:['#f8ddc4','#efc6a3','#deac86','#c9946a','#b77c53','#a36b48','#8b563a','#75432f','#5b3428','#42271f'],
  hair:['#241b18','#3d2b24','#634332','#8d613d','#ba8b54','#dfc083','#eee1bd','#a94b38','#747875','#ece9e1'],
  eyes:['#34403a','#476775','#739aa5','#577451','#8b6d3b','#b59b62','#674b3b','#383542'],
  top:['#2b6777','#52ab98','#334d60','#546576','#8f6462','#a77e54','#d6b56d','#e5e3db','#34383c','#5d506f'],
  bg:['#c8d8e4','#e0ebe9','#f2f2f2','#f1e5d6','#e6d9dc','#cad6ca','#d9d3e1','#2b6777'],
  hatColor:['#2b6777','#35434d','#b3634f','#c3a36b','#e6dfce','#493c54'],
  glassesColor:['#31434a','#664a39','#ac7847','#bfc8cc','#74464b','#ece9e1'],
  necklaceColor:['#d4ad62','#d3d8dc','#b07b56','#f2efe5'], earringColor:['#d4ad62','#d3d8dc','#b07b56','#f2efe5'],
  lipColor:['#b96669','#ad5a50','#c98e87','#824c55','#ca7d75'],
};
function fullHex(value, fallback) { return /^#[a-f0-9]{6}$/i.test(value || '') ? value : /^#[a-f0-9]{3}$/i.test(value || '') ? '#'+[...value.slice(1)].map(part=>part+part).join('') : fallback; }

function Colour({ field, label, ch, custom, apply, remember, t }) {
    const values = [...new Set([...(colours[field] || []),...(custom[field] || [])])].filter(value=>fullHex(value,null));
    const current = fullHex(ch[field],avatarDefaults[field]);
    return <fieldset className="avatar-control"><legend>{t(label)}</legend><div className="avatar-colours">
      {values.map(value=><button key={value} type="button" className="avatar-colour" aria-label={`${t(label)} ${value}`} aria-pressed={current.toLowerCase()===fullHex(value,value).toLowerCase()} style={{'--swatch':value}} onClick={()=>apply(field,value)} />)}
      <label className="avatar-custom-colour"><span>{t('Custom colour')}</span><input aria-label={`${t('Custom colour')}: ${t(label)}`} type="color" value={current} onChange={event=>apply(field,event.target.value)} onBlur={event=>remember(field,event.target.value)} /></label>
    </div></fieldset>;
  }
function Options({ field,label, ch, apply, t }) {
    return <fieldset className="avatar-control"><legend>{t(label)}</legend><div className="avatar-options">
      {choices[field].map((name,index)=><button key={name} type="button" aria-label={t(name)} aria-pressed={Number(ch[field] || 0)===index} onClick={()=>apply(field,index)}><Avatar character={{...ch,[field]:index}} size={58} label={t(name)} /><span>{t(name)}</span></button>)}
    </div></fieldset>;
  }
function Toggle({ field,label, ch, apply, t }) { return <label className="avatar-toggle"><input type="checkbox" checked={Boolean(ch[field])} onChange={event=>apply(field,event.target.checked)}/><span>{t(label)}</span></label>; }

export default function AvatarEditor({ character, onChange, onClose }) {
  const { t } = useLocale();
  const [category, setCategory] = useState('Face');
  const [custom, setCustom] = useState(() => { try { return JSON.parse(localStorage.getItem('classio_custom_colors_v1') || '{}'); } catch { return {}; } });
  const id = useId();
  const ch = { ...avatarDefaults, ...character };
  function apply(field,value) { onChange({ ...ch, [field]:value }); }
  function remember(field,value) {
    const next = { ...custom, [field]: [...new Set([...(custom[field] || []),value])].slice(-10) };
    setCustom(next); try { localStorage.setItem('classio_custom_colors_v1',JSON.stringify(next)); } catch {}
  }
  const tabs=['Face','Hair','Eyes','Expression','Outfit','Accessories'];
  return <Dialog label={t('Your Classio character')} onClose={onClose} className="avatar-dialog">
    <header className="avatar-editor-heading"><p className="entry-kicker">{t('Make it yours')}</p><h2>{t('Your Classio character')}</h2><p>{t('A familiar face for your workspace.')}</p></header>
    <div className="avatar-editor-layout">
      <aside className="avatar-stage"><Avatar character={ch} size={240} animate label={t('Your avatar')}/><label>{t('Nickname')}<input aria-label={t('Nickname')} value={ch.name || ''} maxLength={40} onChange={event=>apply('name',event.target.value)} /></label><p>{t('Changes appear immediately.')}</p></aside>
      <div className="avatar-customize">
        <div role="tablist" aria-label={t('Your avatar')} className="avatar-tabs" onKeyDown={event=>{
          if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
          event.preventDefault();const current=tabs.indexOf(category);const next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(current+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
          setCategory(tabs[next]);event.currentTarget.querySelectorAll('[role=tab]')[next]?.focus();
        }}>{tabs.map(name=><button key={name} id={id+name} role="tab" type="button" aria-selected={category===name} tabIndex={category===name?0:-1} aria-controls={id+'panel'} onClick={()=>setCategory(name)}>{t(name)}</button>)}</div>
        <div role="tabpanel" id={id+'panel'} aria-labelledby={id+category} className="avatar-tab-panel">
          {category==='Face' && <><Options ch={ch} apply={apply} t={t} field="faceShape" label="Face shape"/><Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="skin" label="Skin tone"/><Options ch={ch} apply={apply} t={t} field="nose" label="Nose"/><Toggle ch={ch} apply={apply} t={t} field="blush" label="Blush"/><Toggle ch={ch} apply={apply} t={t} field="freckles" label="Freckles"/></>}
          {category==='Hair' && <><Options ch={ch} apply={apply} t={t} field="hairStyle" label="Hair style"/><Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="hair" label="Hair colour"/></>}
          {category==='Eyes' && <><Options ch={ch} apply={apply} t={t} field="eyeShape" label="Eye shape"/><Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="eyes" label="Eye colour"/><Options ch={ch} apply={apply} t={t} field="eyebrow" label="Eyebrows"/><label className="avatar-range">{t('Eye spacing')}<input aria-label={t('Eye spacing')} type="range" min="-6" max="6" value={ch.eyeSpacing || 0} onChange={event=>apply('eyeSpacing',Number(event.target.value))}/></label></>}
          {category==='Expression' && <><Options ch={ch} apply={apply} t={t} field="mouth" label="Expression"/><label className="avatar-range">{t('Smile position')}<input aria-label={t('Smile position')} type="range" min="-4" max="4" value={ch.mouthPosition || 0} onChange={event=>apply('mouthPosition',Number(event.target.value))}/></label><Options ch={ch} apply={apply} t={t} field="facialHair" label="Facial hair"/><Toggle ch={ch} apply={apply} t={t} field="lips" label="Lip gloss"/>{ch.lips && <Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="lipColor" label="Lip colour"/>}</>}
          {category==='Outfit' && <><Options ch={ch} apply={apply} t={t} field="topStyle" label="Outfit style"/><Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="top" label="Outfit colour"/><Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="bg" label="Background"/></>}
          {category==='Accessories' && <><Options ch={ch} apply={apply} t={t} field="hat" label="Hat"/>{Number(ch.hat)>0 && <Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="hatColor" label="Hat colour"/>}<Options ch={ch} apply={apply} t={t} field="glasses" label="Glasses"/>{Number(ch.glasses)>0 && <Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="glassesColor" label="Glasses colour"/>}<Options ch={ch} apply={apply} t={t} field="earring" label="Earrings"/>{Number(ch.earring)>0 && <Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="earringColor" label="Jewellery colour"/>}<Options ch={ch} apply={apply} t={t} field="necklace" label="Necklace"/>{Number(ch.necklace)>0 && <Colour ch={ch} custom={custom} apply={apply} remember={remember} t={t} field="necklaceColor" label="Jewellery colour"/>}</>}
        </div>
      </div>
    </div>
    <footer className="avatar-editor-footer"><button type="button" className="avatar-reset" onClick={()=>{if(window.confirm(t('Reset appearance')+'?'))onChange({...ch,...avatarDefaults,name:ch.name});}}>{t('Reset appearance')}</button><button type="button" className="avatar-done" onClick={onClose}>{t('Done')}</button></footer>
  </Dialog>;
}
