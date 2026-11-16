import {DeckSlide,deckDesigns} from './DeckDesign';
import {DECK_THEMES,SLIDE_TYPES} from './presentationModel';
import UiIcon from './UiIcon';

export function OptionCards({label,value,options,onChange}){
 return <fieldset className="presentation-options"><legend>{label}</legend><div className="presentation-option-grid">{options.map(option=><button type="button" key={option.id} aria-label={label==='Layout'?'Layout: '+option.name:option.name} aria-pressed={value===option.id} onClick={()=>onChange(option.id)}><span className={'option-mini option-'+option.id} aria-hidden="true"><i/><i/><i/></span><strong>{option.name}{value===option.id&&<UiIcon name="check" size={15}/>}</strong>{option.description&&<small>{option.description}</small>}</button>)}</div></fieldset>;
}
export function LayoutCards({value,onChange}){return <OptionCards label="Layout" value={value} options={deckDesigns} onChange={onChange}/>;}
export function SlideTypeCards({onChoose}){return <OptionCards label="Add slide" options={SLIDE_TYPES.map(([id,name,description])=>({id,name,description}))} onChange={onChoose}/>;}
export function PlacementCards({value,onChange}){return <OptionCards label="Image placement" value={value||'auto'} options={[['auto','Automatic'],['left','Left'],['right','Right'],['center','Center'],['full','Full slide'],['background','Background']].map(([id,name])=>({id,name}))} onChange={onChange}/>;}
export function ThemeCards({value,onChange,custom}){
 return <fieldset className="presentation-options theme-options"><legend>Theme</legend><div className="presentation-theme-grid">{[...DECK_THEMES,...(custom?[custom]:[])].map(theme=><button type="button" key={theme.id} aria-label={'Theme: '+theme.label} aria-pressed={value===theme.id} onClick={()=>onChange(theme.id)}><span style={{background:theme.bg,color:theme.text,borderColor:theme.accent}}><b>A clear idea.</b><i style={{background:theme.accent}}/></span><strong>{theme.label}{value===theme.id&&<UiIcon name="check" size={14}/>}</strong></button>)}</div></fieldset>;
}
export function DesignSample({theme,layout}){return <DeckSlide slide={{type:'content',title:'Ideas worth sharing',bullets:['Explain clearly','Show the evidence']}} theme={theme} layout={layout} index={0} total={1}/>;}
