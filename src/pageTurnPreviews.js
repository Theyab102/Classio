import {drawInkPreview} from './DocumentInk';
import {SourceSlidePreview} from './PowerPointViewer';

function addInk(node,strokes,width,height){
  if(!strokes?.length)return;
  const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(width));canvas.height=Math.max(1,Math.round(height));
  canvas.className='page-turn-ink-copy';drawInkPreview(canvas.getContext('2d'),strokes,canvas.width,canvas.height);node.append(canvas);
}

export function pdfPagePreview(pdf,page,scale,strokes){
  let task,cancelled=false;
  const promise=(async()=>{
    const source=await pdf.getPage(page);if(cancelled)throw new Error('Cancelled preview');
    const viewport=source.getViewport({scale}),ratio=Math.min(window.devicePixelRatio||1,2,2400/Math.max(viewport.width,viewport.height));
    const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(viewport.width*ratio));canvas.height=Math.max(1,Math.round(viewport.height*ratio));
    task=source.render({canvasContext:canvas.getContext('2d'),viewport,transform:[ratio,0,0,ratio,0,0]});await task.promise;if(cancelled)throw new Error('Cancelled preview');
    const node=document.createElement('div');node.className='pdf-page-frame';canvas.className='pdf-original-canvas';node.append(canvas);addInk(node,strokes,canvas.width,canvas.height);
    return{node,width:viewport.width,height:viewport.height};
  })();
  return{promise,cancel(){cancelled=true;task?.cancel();}};
}

export async function powerPointPagePreview(slide,ratio,strokes,width){
  const{renderToStaticMarkup}=await import('react-dom/server');
  const node=document.createElement('div');node.className='presentation-source-size';
  node.innerHTML=renderToStaticMarkup(<section className="document-viewer ppt-document"><SourceSlidePreview slide={slide} width={ratio} height={1} positioned={slide.items.every(item=>item.position)}/></section>);
  addInk(node.querySelector('.ppt-page'),strokes,1280,1280/ratio);
  await Promise.all([...node.querySelectorAll('img')].map(image=>image.decode().catch(()=>{})));
  return{node,width,height:width/ratio};
}

// Canvas cloneNode does not copy its bitmap. Keep the real page and its ink in
// the outgoing snapshot while React renders the next page underneath it.
export function copyPageSnapshot(source){
  const copy=source.cloneNode(true),originals=source.querySelectorAll('canvas'),canvases=copy.querySelectorAll('canvas');
  originals.forEach((canvas,index)=>{canvases[index].getContext('2d').drawImage(canvas,0,0);});
  for(const node of [copy,...copy.querySelectorAll('*')]){
    for(const name of node.getAttributeNames())if(name==='id'||name==='role'||name==='tabindex'||name.startsWith('aria-')||name==='data-page-turn-current')node.removeAttribute(name);
    if(node.classList.contains('studio-ink')){node.classList.remove('studio-ink');node.classList.add('page-turn-ink-copy');}
  }
  copy.style.visibility='visible';return copy;
}
