import {normalizeMathSource} from './mathSyntax';
import {normalizeLatex} from './mathMarkdown';
const escape=value=>String(value).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
let converter;
async function mathConverter(){
  if(!converter)converter=Promise.all([import('mathjax-full/js/mathjax.js'),import('mathjax-full/js/input/tex.js'),import('mathjax-full/js/output/svg.js'),import('mathjax-full/js/adaptors/liteAdaptor.js'),import('mathjax-full/js/handlers/html.js'),import('mathjax-full/js/input/tex/AllPackages.js')]).then(([core,input,output,adapter,handler,packages])=>{const adaptor=adapter.liteAdaptor();handler.RegisterHTMLHandler(adaptor);const document=core.mathjax.document('',{InputJax:new input.TeX({packages:packages.AllPackages}),OutputJax:new output.SVG({fontCache:'none'})});return{adaptor,document};}).catch(error=>{converter=null;throw error;});
  return converter;
}
export function mathSegments(source){const text=normalizeMathSource(source),parts=[],regex=/\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g;let match,start=0;while((match=regex.exec(text))){if(match.index>start)parts.push({text:text.slice(start,match.index)});parts.push({latex:normalizeLatex(match[1]||match[2]||match[3]||match[4])});start=regex.lastIndex;}if(start<text.length)parts.push({text:text.slice(start)});return parts;}
export const containsExportMath=text=>mathSegments(String(text||'')).some(part=>part.latex!==undefined);
export async function richTextSvg(text,{width,fontSize=21,color='253648',fontFace='Arial',bold=false}){
  const pixels=width*96,font=fontSize*96/72,parts=mathSegments(text),canvas=document.createElement('canvas'),context=canvas.getContext('2d');context.font=(bold?'700 ':'400 ')+font+'px '+fontFace;
  const tokens=[];
  for(const part of parts){if(part.latex!==undefined){const{adaptor,document:math}=await mathConverter(),node=math.convert(part.latex,{display:false,em:font,ex:font*.5,containerWidth:pixels});const svg=adaptor.firstChild(node),markup=adaptor.outerHTML(svg);if(/data-mjx-error|merror/.test(markup))throw new Error('An equation needs review before this presentation can be exported.');const w=Number(adaptor.getAttribute(svg,'width')?.replace('ex',''))*font*.5,h=Number(adaptor.getAttribute(svg,'height')?.replace('ex',''))*font*.5;if(!w||!h)throw new Error('Could not size an equation.');const align=/vertical-align:\s*([-.\d]+)ex/.exec(adaptor.getAttribute(svg,'style')||''),depth=align?Math.max(0,-Number(align[1])*font*.5):font*.2,scale=Math.min(1,pixels/w);tokens.push({width:w*scale,ascent:(h-depth)*scale,descent:depth*scale,svg:markup.replace(/currentColor/g,'#'+color),height:h*scale});}
    else for(const word of part.text.replace(/\*\*|__/g,'').split(/(\s+)/)){if(!word)continue;tokens.push({text:word,width:context.measureText(word).width,ascent:font*.9,descent:font*.3,newline:word.includes('\n')});}}
  const lines=[];let current=[],used=0;function flush(){if(current.length){lines.push(current);current=[];used=0;}}
  for(const token of tokens){if(token.newline){flush();continue;}if(used+token.width>pixels&&current.length)flush();if(!current.length&&/^\s+$/.test(token.text||''))continue;current.push(token);used+=token.width;}flush();
  let y=2,content='';for(const line of lines){const ascent=Math.max(...line.map(t=>t.ascent)),descent=Math.max(...line.map(t=>t.descent));let x=0;for(const token of line){if(token.svg){const inner=token.svg.replace(/<svg\b[^>]*>/,'').replace(/<\/svg>\s*$/,''),view=/viewBox="([^"]+)"/.exec(token.svg)?.[1];content+=`<svg x="${x}" y="${y+ascent-token.ascent}" width="${token.width}" height="${token.height}" viewBox="${view}">${inner}</svg>`;}else content+=`<text x="${x}" y="${y+ascent}" fill="#${color}" font-family="${escape(fontFace)}" font-size="${font}" font-weight="${bold?'700':'400'}" xml:space="preserve">${escape(token.text)}</text>`;x+=token.width;}y+=ascent+descent+font*.22;}
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${Math.ceil(pixels)}" height="${Math.ceil(y+2)}" viewBox="0 0 ${Math.ceil(pixels)} ${Math.ceil(y+2)}">${content}</svg>`;
  return{data:'data:image/svg+xml;base64,'+btoa(unescape(encodeURIComponent(svg))),height:(y+2)/96,width};
}
export async function addRichText(slide,text,options){
  if(!containsExportMath(text)){slide.addText(String(text||''),options);return;}
  const rendered=await richTextSvg(String(text||''),{width:options.w,fontSize:options.fontSize,color:options.color,fontFace:options.fontFace,bold:options.bold}),scale=Math.min(1,options.h/rendered.height);
  slide.addImage({data:rendered.data,x:options.x,y:options.y,w:options.w*scale,h:rendered.height*scale});
}
