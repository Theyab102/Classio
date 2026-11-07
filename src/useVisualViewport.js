import {useLayoutEffect} from 'react';
// Safari's keyboard resizes the visual viewport without reliably changing vh.
export default function useVisualViewport(ref){
 useLayoutEffect(()=>{
  const view=window.visualViewport;let largest=window.innerHeight,lastWidth=window.innerWidth,frame;
  const update=()=>{const node=ref.current;if(!node)return;const height=view?.height||window.innerHeight,top=view?.offsetTop||0;if(Math.abs(window.innerWidth-lastWidth)>80){largest=window.innerHeight;lastWidth=window.innerWidth;}largest=Math.max(largest,window.innerHeight,height);const editing=/^(INPUT|TEXTAREA)$/.test(document.activeElement?.tagName||'');node.style.setProperty('--visual-height',height+'px');node.style.setProperty('--visual-top',top+'px');node.dataset.keyboardOpen=String(editing&&largest-height>140);};
  const schedule=()=>{cancelAnimationFrame(frame);frame=requestAnimationFrame(update);};update();
  view?.addEventListener('resize',update);view?.addEventListener('scroll',update);window.addEventListener('resize',update);document.addEventListener('focusin',schedule);document.addEventListener('focusout',schedule);
  return()=>{cancelAnimationFrame(frame);view?.removeEventListener('resize',update);view?.removeEventListener('scroll',update);window.removeEventListener('resize',update);document.removeEventListener('focusin',schedule);document.removeEventListener('focusout',schedule);};
 },[ref]);
}
