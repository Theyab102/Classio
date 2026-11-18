import {useLayoutEffect,useRef} from 'react';

// Touch navigates; pen/mouse events belong to InkLayer. Native, non-passive
// listeners are needed on iPad to stop a horizontal swipe becoming browser pan.
export default function useDocumentGestures(viewport,onPage,scope) {
  const callback=useRef(onPage);callback.current=onPage;
  useLayoutEffect(()=>{
    const node=viewport.current;if(!node)return;
    let gesture=null;
    const center=touches=>({x:Array.from(touches).reduce((n,t)=>n+t.clientX,0)/touches.length,y:Array.from(touches).reduce((n,t)=>n+t.clientY,0)/touches.length});
    const distance=touches=>touches.length===2?Math.hypot(touches[1].clientX-touches[0].clientX,touches[1].clientY-touches[0].clientY):0;
    function start(event){
      if(event.target.closest('button,input,textarea,select,[contenteditable],[role=combobox]'))return;
      if(event.touches.length>2){gesture=null;return;}
      const point=center(event.touches);
      gesture={...point,startX:point.x,startY:point.y,count:event.touches.length,axis:null,panned:false,distance:distance(event.touches),fingerInk:!!event.target.closest('.studio-ink[data-finger-drawing=true]'),target:event.target};
    }
    function scroll(delta,axis){
      let remaining=delta,current=gesture.target;
      while(current&&Math.abs(remaining)>.1){
        const root=current===document.body||current===document.documentElement;
        const element=root?document.scrollingElement:current;
        const property=axis==='y'?'scrollTop':'scrollLeft';
        const style=getComputedStyle(element),overflow=style[axis==='y'?'overflowY':'overflowX']||style.overflow;
        if(root||/(auto|scroll)/.test(overflow)){
          const before=element[property];element[property]+=remaining;remaining-=element[property]-before;
        }
        if(root)break;current=current.parentElement;
      }
    }
    function move(event){
      if(!gesture||event.touches.length!==gesture.count)return;
      if(gesture.fingerInk&&gesture.count===1)return;
      const point=center(event.touches),dx=point.x-gesture.startX,dy=point.y-gesture.startY;
      // Leave a spreading/pinching pair to the browser; it must never flip a page.
      if(gesture.distance&&Math.abs(distance(event.touches)-gesture.distance)>Math.max(18,gesture.distance*.15)){gesture=null;return;}
      if(!gesture.axis&&Math.max(Math.abs(dx),Math.abs(dy))>8)gesture.axis=Math.abs(dx)>Math.abs(dy)*1.5?'x':'y';
      if(!gesture.axis)return;
      if(event.cancelable)event.preventDefault();
      if(gesture.axis==='y')scroll(gesture.y-point.y,'y');
      else if(node.scrollWidth>node.clientWidth+2){
        const delta=gesture.x-point.x,before=node.scrollLeft;node.scrollLeft+=delta;
        gesture.panned ||= Math.abs(node.scrollLeft-before)>.1;
      }
      gesture.x=point.x;gesture.y=point.y;
    }
    function end(event){
      if(!gesture)return;
      // Commit only once, when the original group starts lifting. A remaining
      // finger cannot trigger another flip or finish an abandoned ink stroke.
      const session=gesture;gesture=null;
      if(session.fingerInk&&session.count===1)return;
      const dx=session.x-session.startX,dy=session.y-session.startY;
      if(session.axis==='x'&&!session.panned&&Math.abs(dx)>55&&Math.abs(dx)>Math.abs(dy)*1.5){if(event.cancelable)event.preventDefault();callback.current(dx<0?1:-1);}
    }
    const cancel=()=>{gesture=null;};
    node.addEventListener('touchstart',start,{passive:true});node.addEventListener('touchmove',move,{passive:false});node.addEventListener('touchend',end,{passive:false});node.addEventListener('touchcancel',cancel);
    return()=>{node.removeEventListener('touchstart',start);node.removeEventListener('touchmove',move);node.removeEventListener('touchend',end);node.removeEventListener('touchcancel',cancel);};
  },[viewport,scope]);
}
