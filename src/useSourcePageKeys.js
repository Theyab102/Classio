import {useEffect,useRef} from 'react';

// Reading shortcuts work after choosing Source in the sidebar, while controls
// that use arrows themselves (editors, menus, sliders) keep their own keys.
export default function useSourcePageKeys(onPage,enabled=true) {
  const latest=useRef(onPage);latest.current=onPage;
  useEffect(()=>{
    if(!enabled)return;
    const keys=event=>{
      if(event.defaultPrevented||event.altKey||event.ctrlKey||event.metaKey||event.shiftKey||!['ArrowLeft','ArrowRight'].includes(event.key))return;
      if(document.querySelector('[role=dialog][aria-modal=true],.studio-dialog,.command-search-dialog,.phone-navigation-overlay'))return;
      if(event.target.closest?.('input,textarea,select,[contenteditable=true],[role=combobox],[role=slider],[role=menu],.assistant-panel'))return;
      event.preventDefault();latest.current(event.key==='ArrowRight'?1:-1);
    };
    window.addEventListener('keydown',keys);
    return()=>window.removeEventListener('keydown',keys);
  },[enabled]);
}
