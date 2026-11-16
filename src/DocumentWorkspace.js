import {useEffect,useRef} from 'react';
import useStudyFullscreen,{useSlideGestures} from './useStudyFullscreen';
import ChoiceMenu from "./ChoiceMenu";
import { useState } from "react";
import LocalPdfViewer from "./LocalPdfViewer";
import PowerPointViewer from "./PowerPointViewer";
import { useLearningSession, useStudyActivity } from "./LearningSession";
import { downloadStudyFile } from "./studyTools";
import UiIcon from "./UiIcon";

export default function DocumentWorkspace({ file, fileObj, onUpdate, type, children }) {
  const [page, setPage] = useState(file.pptPage || 1), [total, setTotal] = useState(0), [slides, setSlides] = useState([]), [zoom, setZoom] = useState(file.pptZoom || "fit"), [query, setQuery] = useState("");
  const { ask } = useLearningSession();
  const fullscreen=useStudyFullscreen(),viewport=useRef(null),[ratio,setRatio]=useState(16/9);
  useEffect(()=>{const node=viewport.current;if(!node)return;const fit=()=>{const tools=node.querySelector(".document-toolbar")?.getBoundingClientRect().height||64;node.style.setProperty("--fit-slide-width",Math.max(120,Math.min(node.clientWidth-32,(node.clientHeight-tools-32)*ratio))+"px");};fit();const observer=new ResizeObserver(fit);observer.observe(node);return()=>observer.disconnect();},[ratio,fullscreen.full,total]);
  const text = type === "pptx" ? slides[page - 1]?.texts?.join("\n") || "" : "";
  useStudyActivity(type === "pptx" ? file.id : null, "Document source", text, `Slide ${page}`);
  const matches = query.trim().length >= 2 ? slides.map((slide, i) => ({ page: i + 1, text: slide.texts.join(" ") })).filter(slide => slide.text.toLowerCase().includes(query.toLowerCase())) : [];
  const move = next => { const value = Math.max(1, Math.min(total || 1, next)); setPage(value); onUpdate({ ...file, pptPage: value }); };
  const gestures=useSlideGestures(delta=>move(page+delta));
  useEffect(()=>{if(type!=="pptx")return;const keys=event=>{if(document.querySelector(".studio-dialog,.assistant-panel[aria-modal=true],.command-search-dialog")||event.target.closest?.("input,textarea,select,[contenteditable],.assistant-panel,.classio-navigation"))return;if(event.key==="ArrowRight"||event.key==="ArrowLeft"){event.preventDefault();move(page+(event.key==="ArrowRight"?1:-1));}};window.addEventListener("keydown",keys);return()=>window.removeEventListener("keydown",keys);});
  const explain = async (_read, label) => {if(fullscreen.full)await fullscreen.toggle();ask(`Explain ${label || "the current source"} of this document with a worked example.`);};
  return <section ref={fullscreen.ref} className={`source-workspace ${fullscreen.full?"source-is-fullscreen":""}`} aria-label="Source document"><header className="source-heading"><span className="source-heading-icon"><UiIcon name="source" size={24} /></span><div><p className="eyebrow">ORIGINAL MATERIAL</p><h2>Source</h2><p>{fileObj.name}</p></div><button className="quiet-button" onClick={() => downloadStudyFile(fileObj, fileObj.name)}><UiIcon name="download" /><span>Download original</span></button></header>
    {type === "pdf" ? <LocalPdfViewer file={file} fileObj={fileObj} onUpdate={onUpdate} explain={explain} />
      : type === "pptx" ? <><div className="document-toolbar studio-controls"><button className="quiet-button" aria-label="Previous slide" disabled={page <= 1} onClick={() => move(page - 1)}><UiIcon name="back" /></button><label>Slide<input aria-label="Slide number" type="number" min={1} max={total || 1} value={page} onChange={event => move(Number(event.target.value) || 1)} /></label><span>of {total || "…"}</span><button className="quiet-button" aria-label="Next slide" disabled={page >= total} onClick={() => move(page + 1)}><UiIcon name="arrow" /></button><label>Zoom<ChoiceMenu aria-label="PowerPoint zoom" value={zoom} onChange={event => { setZoom(event.target.value); onUpdate({ ...file, pptZoom: event.target.value }); }}><option value="fit">Fit slide</option>{[50, 75, 100, 125, 150, 200].map(n => <option key={n} value={n}>{n}%</option>)}</ChoiceMenu></label><button className="quiet-button" aria-label={fullscreen.full?"Exit source full screen":"Full screen source"} onClick={()=>{setZoom("fit");fullscreen.toggle();}}><UiIcon name={fullscreen.full?"minimize":"expand"}/><span>{fullscreen.full?"Exit full screen":"Full screen"}</span></button><label className="document-search"><UiIcon name="search" /><input type="search" aria-label="Search presentation" value={query} placeholder="Find in slides" onChange={event => setQuery(event.target.value)} /></label><button className="quiet-button" onClick={() => explain(null, `slide ${page}`)}><UiIcon name="generate" />Explain slide</button></div>
        {query.trim().length >= 2 && <div className="document-search-results" aria-live="polite"><span>{matches.length} matching {matches.length === 1 ? "slide" : "slides"}</span>{matches.map(match => <button key={match.page} className="text-button" onClick={() => move(match.page)}>Slide {match.page}</button>)}</div>}
        <div ref={viewport} className={`presentation-source-scroll ${zoom==="fit"?"is-fit-slide":""}`} {...gestures}><div className="presentation-source-size" style={{ width: zoom === "fit" ? "var(--fit-slide-width,100%)" : `${Number(zoom)}%`, minWidth: zoom === "fit" ? 0 : `${Number(zoom)}%` }}><PowerPointViewer file={file} fileObj={fileObj} page={page} onUpdate={onUpdate} onTotalPages={value => { setTotal(value); if (page > value) setPage(1); }} onSlidesLoaded={setSlides} onDimensions={(width,height)=>setRatio(width/height)} /></div></div></>
        : <div className="generic-source-scroll">{children}</div>}
  </section>;
}
