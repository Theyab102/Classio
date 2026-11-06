import { useMemo } from 'react';
import { Marked } from 'marked';
import DOMPurify from 'dompurify';
import katex from 'katex';
import { escapeHtml } from './safeContent';

function mathHtml(latex, displayMode) {
  return katex.renderToString(latex, { displayMode, throwOnError:false, trust:false, maxExpand:1000, maxSize:20 });
}
const markdown = new Marked({gfm:true,breaks:true,renderer:{
  // Model output cannot inject HTML or load remote image trackers.
  html:({text})=>escapeHtml(text),
  image:({text})=>escapeHtml(text||'Image'),
},extensions:[{
  name:'displayMath',level:'block',start:source=>source.search(/\$\$|\\\[/),
  tokenizer(source){const match=/^(?:\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\])(?:\n|$)/.exec(source);if(match)return{type:'displayMath',raw:match[0],latex:match[1]||match[2]};},
  renderer:token=>mathHtml(token.latex,true),
},{
  name:'inlineMath',level:'inline',start:source=>source.search(/\$|\\\(/),
  tokenizer(source){const match=/^(?:\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$|\\\(([\s\S]+?)\\\))/.exec(source);if(match)return{type:'inlineMath',raw:match[0],latex:match[1]||match[2]||match[3],display:!!match[1]};},
  renderer:token=>mathHtml(token.latex,token.display),
}]});

export function studyMarkdownHtml(text) {
  const html=DOMPurify.sanitize(markdown.parse(text||''),{
    USE_PROFILES:{html:true,mathMl:true},FORBID_TAGS:['img','iframe','style','form','input','button','video','audio'],
    ALLOW_DATA_ATTR:false,
  });
  // Links open only after a deliberate click, without sending a referrer.
  const template=document.createElement('template');template.innerHTML=html;
  for(const link of template.content.querySelectorAll('a')){link.rel='noreferrer noopener';link.target='_blank';}
  return template.innerHTML;
}

export default function StudyMarkdown({text,Diagram}) {
  const parts=useMemo(()=>{
    const result=[];let cursor=0;
    const pattern=/^```mermaid[^\S\n]*\n([\s\S]*?)^```[^\S\n]*$/gm;
    for(const match of (text||'').matchAll(pattern)){
      if(match.index>cursor)result.push({html:studyMarkdownHtml(text.slice(cursor,match.index))});
      result.push(Diagram?{diagram:match[1]}:{html:studyMarkdownHtml(match[0])});cursor=match.index+match[0].length;
    }
    if(cursor<(text||'').length)result.push({html:studyMarkdownHtml(text.slice(cursor))});
    return result;
  },[text,Diagram]);
  return <div className="study-markdown">{parts.map((part,index)=>part.diagram?<Diagram key={index} code={part.diagram}/>:<div key={index} dangerouslySetInnerHTML={{__html:part.html}}/>)}</div>;
}
