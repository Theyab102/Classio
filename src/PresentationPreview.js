import useStudyFullscreen,{useSlideGestures} from './useStudyFullscreen';
import {useState} from 'react';
import StudioDialog from './StudioDialog';
import {DeckSlide} from './DeckDesign';
import UiIcon from './UiIcon';

export default function PresentationPreview({slides,theme,layout,design,initialIndex=0,onClose,onEdit,onMove,onDuplicate,onDelete,onAdd,onDesign}){
 const [active,setActive]=useState(initialIndex),fullscreen=useStudyFullscreen(),stage=fullscreen.ref,presenting=fullscreen.full;
 const index=Math.max(0,Math.min(active,slides.length-1)),slide=slides[index];
 const move=delta=>setActive(Math.max(0,Math.min(slides.length-1,index+delta)));
 const gestures=useSlideGestures(move);
 if(!slide)return null;
 return <StudioDialog title="Presentation preview" wide onClose={onClose}><div className="presentation-preview" onKeyDown={event=>{if(event.target.closest('input,textarea,select'))return;if(event.key==='ArrowRight'){event.preventDefault();move(1);}if(event.key==='ArrowLeft'){event.preventDefault();move(-1);}}}>
  <div className="presentation-preview-actions"><button className="quiet-button is-primary" onClick={()=>onEdit(index)}><UiIcon name="edit"/>Edit slide</button><button className="quiet-button" onClick={onDesign}>Theme & layout</button><button className="quiet-button" onClick={onAdd}><UiIcon name="plus"/>Add slide</button><button className="quiet-button" onClick={()=>onDuplicate(index)}>Duplicate</button><button className="quiet-button" disabled={!index} onClick={()=>{onMove(index,-1);move(-1);}}>Move up</button><button className="quiet-button" disabled={index===slides.length-1} onClick={()=>{onMove(index,1);move(1);}}>Move down</button><button className="quiet-button is-danger" onClick={()=>{if(window.confirm('Delete this slide?'))onDelete(index);}}><UiIcon name="trash"/>Delete</button></div>
  <div ref={stage} className={`presentation-stage ${presenting?"is-presenting":""}`} tabIndex={0} {...gestures}><DeckSlide slide={slide} theme={theme} layout={layout} design={design} index={index} total={slides.length}/><div className="presentation-navigation"><button className="quiet-button" aria-label="Previous preview slide" disabled={!index} onClick={()=>move(-1)}><UiIcon name="back"/>Previous</button><span aria-live="polite">{index+1} / {slides.length}</span><button className="quiet-button" aria-label="Next preview slide" disabled={index===slides.length-1} onClick={()=>move(1)}>Next<UiIcon name="arrow"/></button><button className="quiet-button" onClick={fullscreen.toggle}>{presenting?"Exit full screen":"Present full screen"}</button></div></div>
  <nav className="presentation-filmstrip" aria-label="Select preview slide">{slides.map((item,i)=><button key={item.id||i} aria-label={`Show slide ${i+1}`} aria-current={i===index?'true':undefined} onClick={()=>setActive(i)}><DeckSlide slide={item} theme={theme} layout={layout} design={design} index={i} total={slides.length}/><span>{i+1}. {item.title||item.quote||'Slide'}</span></button>)}</nav>
 </div></StudioDialog>;
}
