import {useCallback,useLayoutEffect,useRef} from 'react';
import {copyPageSnapshot} from './pageTurnPreviews';
import './pageTurn.css';

const duration=280,easing='cubic-bezier(.22,.75,.2,1)';
const reduced=()=>window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
const translate=x=>`translate3d(${x}px,0,0)`;

export default function usePageTurn(options){
  const latest=useRef(options);latest.current=options;
  const active=useRef(null);
  const clear=useCallback(()=>{const stage=active.current;if(!stage)return;active.current=null;stage.animations?.forEach(animation=>animation.cancel());clearTimeout(stage.timeout);stage.layer.remove();stage.view.removeAttribute('data-page-turn');},[]);
  const {viewport,resetKey}=options;
  function place(stage){stage.layer.style.left=stage.view.scrollLeft+'px';stage.layer.style.top=stage.view.scrollTop+'px';}
  function positionCopy(node,geometry){Object.assign(node.style,{position:'absolute',left:geometry.x+'px',top:geometry.y+'px',width:geometry.width+'px',height:geometry.height+'px',minWidth:'0',maxWidth:'none',margin:'0',boxSizing:'border-box'});}
  function addIncoming(stage,preview){
    if(active.current!==stage||!preview||stage.incoming.firstChild)return;
    const node=copyPageSnapshot(preview.node),padding=parseFloat(getComputedStyle(stage.view).paddingTop)||0;
    positionCopy(node,{x:preview.width>stage.width?stage.origin.x:(stage.width-preview.width)/2,y:latest.current.previewAtTop?padding:stage.origin.y,width:preview.width,height:preview.height});stage.incoming.append(node);
  }
  function paint(stage,offset){stage.offset=offset;stage.outgoing.style.transform=translate(offset);stage.incoming.style.transform=translate(stage.direction*stage.width+offset);}
  function create(direction){
    const config=latest.current,view=config.viewport.current,surface=config.surface.current;
    if(!view||!surface||!config.ready||reduced()||!surface.getBoundingClientRect().width)return null;
    const bounds=surface.getBoundingClientRect(),outer=view.getBoundingClientRect(),layer=document.createElement('div');
    const outgoing=document.createElement('div'),incoming=document.createElement('div');layer.className='page-turn-overlay';layer.setAttribute('aria-hidden','true');layer.inert=true;outgoing.className='page-turn-plane page-turn-outgoing';incoming.className='page-turn-plane page-turn-incoming';
    const stage={view,layer,outgoing,incoming,direction,width:view.clientWidth,from:config.page,target:config.page+direction,phase:'dragging',offset:0,origin:{x:bounds.left-outer.left-view.clientLeft,y:bounds.top-outer.top-view.clientTop,width:bounds.width,height:bounds.height}};
    const copy=copyPageSnapshot(surface);positionCopy(copy,stage.origin);outgoing.append(copy);layer.append(outgoing,incoming);
    Object.assign(layer.style,{width:view.clientWidth+'px',height:view.clientHeight+'px',background:getComputedStyle(view).backgroundColor});
    active.current=stage;view.dataset.pageTurn='dragging';place(stage);view.append(layer);paint(stage,0);
    if(stage.target>=1&&stage.target<=config.total)Promise.resolve().then(()=>active.current===stage?config.preview?.(stage.target):null).then(preview=>addIncoming(stage,preview)).catch(()=>{});
    return stage;
  }
  function animate(stage,commit){
    if(active.current!==stage)return;
    if(reduced()||!stage.outgoing.animate){clear();return;}
    stage.phase='animating';stage.view.dataset.pageTurn=commit?'turning':'returning';place(stage);
    const end=commit?-stage.direction*stage.width:0,settings={duration:commit?duration:180,easing,fill:'forwards'};
    stage.animations=[stage.outgoing.animate([{transform:translate(stage.offset)},{transform:translate(end)}],settings),stage.incoming.animate([{transform:translate(stage.direction*stage.width+stage.offset)},{transform:translate(stage.direction*stage.width+end)}],settings)];
    Promise.allSettled(stage.animations.map(animation=>animation.finished)).then(()=>{if(active.current===stage)clear();});
  }
  function cancel(){const stage=active.current;if(stage?.phase==='dragging')animate(stage,false);}
  function drag(offset){
    if(reduced()||!offset)return;
    let stage=active.current;if(stage&&stage.phase!=='dragging')return;
    const direction=offset<0?1:-1;
    if(stage&&stage.direction!==direction){clear();stage=null;}
    stage ||= create(direction);if(!stage)return;
    const boundary=stage.target<1||stage.target>latest.current.total;
    paint(stage,Math.sign(offset)*Math.min(Math.abs(offset)*(boundary ? .2 : 1),stage.width*.95));
  }
  function goTo(next){
    const config=latest.current,requested=Number(next),value=Math.max(1,Math.min(config.total||1,Number.isFinite(requested)?Math.trunc(requested):config.page));
    if(value===config.page){cancel();return;}
    let stage=active.current;
    if(stage&&stage.phase!=='dragging')clear();
    if(!stage||active.current!==stage)stage=create(value>config.page?1:-1);
    if(stage){stage.target=value;stage.phase='waiting';stage.view.dataset.pageTurn='waiting';stage.timeout=setTimeout(()=>{if(active.current===stage)clear();},12000);}
    config.onChange(value);if(stage)place(stage);
  }
  useLayoutEffect(()=>{
    const stage=active.current;if(!stage||stage.phase!=='waiting'||options.page!==stage.target)return;
    if(options.failed){clear();return;}
    if(options.ready){
      stage.incoming.replaceChildren();const surface=options.surface.current,bounds=surface.getBoundingClientRect(),outer=stage.view.getBoundingClientRect();
      const copy=copyPageSnapshot(surface);positionCopy(copy,{x:bounds.left-outer.left-stage.view.clientLeft,y:bounds.top-outer.top-stage.view.clientTop,width:bounds.width,height:bounds.height});stage.incoming.append(copy);clearTimeout(stage.timeout);animate(stage,true);
    }
  });
  useLayoutEffect(()=>{clear();const view=viewport.current;if(!view)return;
    const observer=new ResizeObserver(()=>{const stage=active.current;if(stage&&(Math.abs(view.clientWidth-stage.width)>1||Math.abs(view.clientHeight-parseFloat(stage.layer.style.height))>1))clear();});observer.observe(view);
    const media=window.matchMedia?.('(prefers-reduced-motion: reduce)'),change=()=>{if(media.matches)clear();};media?.addEventListener?.('change',change);
    return()=>{observer.disconnect();media?.removeEventListener?.('change',change);clear();};
  },[viewport,resetKey,clear]);
  return{goTo,gesture:{drag,cancel,start(){if(active.current?.phase==='dragging'||active.current?.view.dataset.pageTurn==='returning')clear();}}};
}
