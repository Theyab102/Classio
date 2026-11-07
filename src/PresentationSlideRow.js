import {DeckSlide} from './DeckDesign';
import StudyMarkdown from './StudyMarkdown';
import UiIcon from './UiIcon';

export default function PresentationSlideRow({slide,index,total,theme,layout,design,editing,onEdit,onMove,onDuplicate,onAnnotate,onDelete}){
 return <header className="deck-card-row">
  <button type="button" className="deck-card-thumbnail" aria-label={`Preview slide ${index+1}`} onClick={onEdit}><DeckSlide slide={slide} theme={theme} layout={layout} design={design} index={index} total={total}/></button>
  <div className="deck-slide-copy"><div className="deck-slide-meta"><span>{slide.type}</span><span>Slide {index+1}</span></div><h3><StudyMarkdown inline text={slide.title||slide.quote||'Slide'}/></h3>{!!slide.bullets?.length&&<p>{slide.bullets.length} bullet{slide.bullets.length===1?'':'s'}</p>}</div>
  <div className="deck-slide-actions" role="toolbar" aria-label={`Slide ${index+1} actions`}>
   <button type="button" className="quiet-button deck-icon-action" aria-label={`Move slide ${index+1} up`} disabled={index===0} onClick={()=>onMove(-1)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 15 6-6 6 6"/></svg></button>
   <button type="button" className="quiet-button deck-icon-action" aria-label={`Move slide ${index+1} down`} disabled={index===total-1} onClick={()=>onMove(1)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></button>
   <button type="button" className="quiet-button deck-icon-action" aria-label={`Duplicate slide ${index+1}`} title="Duplicate" onClick={onDuplicate}><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/></svg></button>
   <button type="button" className="quiet-button" onClick={onEdit}><UiIcon name="edit" size={14}/>{editing?'Close':'Edit'}</button>
   <button type="button" className="quiet-button" onClick={onAnnotate}><UiIcon name="edit" size={14}/>Annotate</button>
   <button type="button" className="quiet-button deck-icon-action is-danger" aria-label={`Delete slide ${index+1}`} onClick={onDelete}><UiIcon name="trash" size={15}/></button>
  </div>
 </header>;
}
