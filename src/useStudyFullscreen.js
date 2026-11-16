import {useEffect,useRef,useState} from 'react';
import {lockBodyScroll} from './bodyScrollLock';
export default function useStudyFullscreen() {
  const ref=useRef(null),[full,setFull]=useState(false);
  useEffect(()=>{if(!full)return;const unlock=lockBodyScroll();const key=event=>{const dialogs=document.querySelectorAll('.studio-dialog'),top=dialogs[dialogs.length-1];if(event.key==='Escape'&&(!top||top.contains(ref.current))){event.preventDefault();if(document.fullscreenElement===ref.current)document.exitFullscreen?.().catch(()=>{});setFull(false);}};window.addEventListener('keydown',key);return()=>{unlock();window.removeEventListener('keydown',key);};},[full]);
  useEffect(()=>{const element=ref.current;const change=()=>{if(!document.fullscreenElement)setFull(false);};document.addEventListener('fullscreenchange',change);return()=>{document.removeEventListener('fullscreenchange',change);if(document.fullscreenElement===element)document.exitFullscreen?.().catch(()=>{});};},[]);
  async function toggle() {
    if(full){if(document.fullscreenElement===ref.current)await document.exitFullscreen?.().catch(()=>{});setFull(false);}
    else{setFull(true);try{await ref.current?.requestFullscreen?.();}catch{/* The in-app full screen remains available on iPad browsers. */}}
  }
  return {ref,full,toggle};
}
export function useSlideGestures(onMove) {
  const start=useRef(null);
  return {
    onPointerDown:event=>{if(event.pointerType==='mouse'||event.target.closest('button,input,textarea,select,[contenteditable],.studio-ink:not([data-tool=pan])'))return;start.current={x:event.clientX,y:event.clientY};},
    onPointerUp:event=>{if(!start.current)return;const dx=event.clientX-start.current.x,dy=event.clientY-start.current.y;start.current=null;if(Math.abs(dx)>55&&Math.abs(dx)>Math.abs(dy)*1.5)onMove(dx<0?1:-1);},
    onPointerCancel:()=>{start.current=null;},
  };
}
