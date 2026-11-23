import {graphSvg} from './mathGraphs';
import { useMemo } from "react";
import { markdownToNoteHtml } from "./noteFormat";
import { typesetMath } from "./mathMarkdown";

export default function StudyMarkdown({ text = "", className = "", allowImages = false, allowLinks = true, inline = false }) {
  const html = useMemo(() => {
    const graphs=[],marker='CLASSIO_GRAPH_'+Math.random().toString(36).slice(2)+'_';
    const markdown=String(text).replace(/```classio-graph\s*\n([\s\S]*?)\n```/g,(_block,json)=>{const index=graphs.length;try{graphs.push(graphSvg(JSON.parse(json)));}catch{graphs.push(null);}return '\n\n'+marker+index+'\n\n';});
    const host = document.createElement("div"); host.innerHTML = markdownToNoteHtml(markdown);
    if (!allowImages) host.querySelectorAll("img").forEach(image => image.replaceWith(image.alt || ""));
    if (!allowLinks) host.querySelectorAll("a").forEach(link => link.replaceWith(...link.childNodes));
    host.querySelectorAll('p').forEach(paragraph=>{const value=paragraph.textContent.trim();if(!value.startsWith(marker))return;const index=Number(value.slice(marker.length));if(!Number.isInteger(index)||index<0||index>=graphs.length)return;if(!graphs[index]){paragraph.textContent='This graph could not be drawn. Check the equation or request a simpler graph.';return;}const graph=document.createElement('figure');graph.className='study-function-graph';graph.innerHTML=graphs[index];paragraph.replaceWith(graph);});
    typesetMath(host);
    if(inline) host.querySelectorAll("p,h1,h2,h3,h4,h5,h6,blockquote,ul,ol,li").forEach(node => { const span=document.createElement("span"); span.append(...node.childNodes); node.replaceWith(span,document.createTextNode(" ")); });
    return host.innerHTML;
  }, [text, allowImages, allowLinks, inline]);
  const Tag=inline?"span":"div";
  return <Tag dir="auto" className={`study-markdown ${inline?"study-inline":""} ${className}`} dangerouslySetInnerHTML={{ __html: html }} />;
}
