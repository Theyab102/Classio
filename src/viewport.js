// Layout only: visual-viewport changes are never persisted or sent anywhere.
let frame;
function update(){
  frame=undefined;
  const height=window.visualViewport?.height||window.innerHeight;
  document.documentElement.style.setProperty('--visible-height',height+'px');
  document.documentElement.style.setProperty('--visible-width',(window.visualViewport?.width||window.innerWidth)+'px');
  document.documentElement.style.setProperty('--viewport-left',(window.visualViewport?.offsetLeft||0)+'px');
  document.documentElement.style.setProperty('--viewport-top',(window.visualViewport?.offsetTop||0)+'px');
  document.documentElement.dataset.shortViewport=height<600?'true':'false';
}
function schedule(){if(frame===undefined)frame=requestAnimationFrame(update);}
update();
window.addEventListener('resize',schedule);
window.visualViewport?.addEventListener('resize',schedule);
window.visualViewport?.addEventListener('scroll',schedule);
