import {useRef,useState} from 'react';
import {DeckSlide} from './DeckDesign';
import {deckGeometry} from './deckGeometry';

export default function SlideCanvasEditor({slide,theme,layout,design,index,total,onChange,onEditText}){
 const surface=useRef(null),drag=useRef(null),[selected,setSelected]=useState(null),[transient,setTransient]=useState(null);
 const shown=transient||slide,geometry=deckGeometry(shown,layout,index),box=selected&&geometry[selected];
 function start(event,resize=false){
   if(event.button!==0)return;event.preventDefault();event.stopPropagation();event.currentTarget.setPointerCapture(event.pointerId);
   drag.current={x:event.clientX,y:event.clientY,box:{...box},resize,slide};
 }
 function move(event){
   const d=drag.current;if(!d)return;const bounds=surface.current.getBoundingClientRect(),dx=(event.clientX-d.x)/bounds.width*13.333,dy=(event.clientY-d.y)/bounds.height*7.5;
   let next=d.resize?{...d.box,w:Math.max(.5,d.box.w+dx),h:Math.max(.3,d.box.h+dy)}:{...d.box,x:d.box.x+dx,y:d.box.y+dy};
   if(d.resize&&selected==='image'&&slide.lockAspect!==false)next.h=next.w*d.box.h/d.box.w;
   setTransient({...slide,geometry:{...slide.geometry,[selected]:next}});
 }
 function end(){if(drag.current&&transient)onChange(transient);drag.current=null;setTransient(null);}
 function keys(event){
   if(!box||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key))return;
   event.preventDefault();const delta=event.shiftKey?.15:.03,next={...box,x:box.x+(event.key==='ArrowRight'?delta:event.key==='ArrowLeft'?-delta:0),y:box.y+(event.key==='ArrowDown'?delta:event.key==='ArrowUp'?-delta:0)};
   onChange({...slide,geometry:{...slide.geometry,[selected]:next}});
 }
 return <div className="slide-canvas-editor" ref={surface} tabIndex={0} aria-label="Interactive slide preview" onKeyDown={keys} onClick={event=>{const key=event.target.closest('[data-slide-element]')?.dataset.slideElement;setSelected(key||null);surface.current.focus({preventScroll:true});}} onDoubleClick={event=>{const key=event.target.closest('[data-slide-element]')?.dataset.slideElement;if(key==='title'||key==='body')onEditText(key);}}>
  <DeckSlide slide={shown} theme={theme} layout={layout} design={design} index={index} total={total}/>
  {box&&<div className="slide-selection" onClick={event=>event.stopPropagation()} style={{left:box.x/13.333*100+'%',top:box.y/7.5*100+'%',width:box.w/13.333*100+'%',height:box.h/7.5*100+'%'}} onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={()=>{drag.current=null;setTransient(null);}}><span>{selected==='image'?'Image':'Text'} · drag to move</span><button type="button" className="slide-resize" aria-label={'Resize '+selected} onPointerDown={event=>start(event,true)} onPointerMove={move} onPointerUp={end}/></div>}
 </div>;
}
