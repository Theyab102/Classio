// Equations are parsed as arithmetic, never evaluated as JavaScript.
const functions={sin:Math.sin,cos:Math.cos,tan:Math.tan,sqrt:Math.sqrt,abs:Math.abs,exp:Math.exp,ln:Math.log,log:Math.log10};
export const GRAPH_OUTPUT_GUIDE='When a function, graph or end behavior is being explained, show its graph using a fenced classio-graph block containing JSON: {"expression":"x^3-3*x","xMin":-4,"xMax":4,"title":"Cubic end behavior"}. Use the actual supplied equation, never an invented replacement. Supported arithmetic: x, numbers, + - * / ^, parentheses, sin, cos, tan, sqrt, abs, exp, ln, log. Write multiplication explicitly. Optional yMin/yMax set the visible range. Explain the graph in ordinary lesson-level language next to it. A graph is drawn locally; do not ask for an image URL.';
export function graphFunction(expression){
 const source=String(expression||'').replace(/\s+/g,'');
 const tokens=source.match(/(?:\d*\.?\d+(?:e[+-]?\d+)?)|[A-Za-z]+|[+\-*/^(),]/g)||[];
 if(!source||source.length>300||tokens.length>100||tokens.join('')!==source)throw new Error('Unsupported graph equation');
 let index=0;
 function parse(min=0){
  const token=tokens[index++];let left;
  if(token==='+'||token==='-'){const next=parse(3);left=x=>(token==='-'?-1:1)*next(x);}
  else if(token==='('){left=parse();if(tokens[index++]!==')')throw new Error('Unbalanced graph equation');}
  else if(token==='x')left=x=>x;
  else if(token==='pi'||token==='e'){const value=token==='pi'?Math.PI:Math.E;left=()=>value;}
  else if(Object.prototype.hasOwnProperty.call(functions,token)){if(tokens[index++]!=='(')throw new Error('Graph functions need parentheses');const argument=parse();if(tokens[index++]!==')')throw new Error('Unbalanced graph function');left=x=>functions[token](argument(x));}
  else if(token&&/^\d*\.?\d+(?:e[+-]?\d+)?$/.test(token)){const number=Number(token);if(!Number.isFinite(number))throw new Error('Invalid graph number');left=()=>number;}
  else throw new Error('Unsupported graph token');
  while(index<tokens.length){const op=tokens[index],precedence={'+':1,'-':1,'*':2,'/':2,'^':4}[op];if(!precedence||precedence<min)break;index++;const right=parse(op==='^'?precedence:precedence+1),previous=left;left=x=>{const a=previous(x),b=right(x);return op==='+'?a+b:op==='-'?a-b:op==='*'?a*b:op==='/'?(b===0?NaN:a/b):Math.pow(a,b);};}
  return left;
 }
 const evaluate=parse();if(index!==tokens.length)throw new Error('Incomplete graph equation');return x=>{const result=evaluate(x);return Number.isFinite(result)&&Math.abs(result)<1e12?result:NaN;};
}
export function sampleGraph(spec){
 if(!spec||typeof spec.expression!=='string')throw new Error('A graph needs an equation');const fn=graphFunction(spec.expression);
 const xMin=Number(spec.xMin??-10),xMax=Number(spec.xMax??10);if(!Number.isFinite(xMin)||!Number.isFinite(xMax)||xMin>=xMax||Math.max(Math.abs(xMin),Math.abs(xMax))>1e6)throw new Error('Invalid graph range');
 const points=Array.from({length:401},(_,i)=>{const x=xMin+(xMax-xMin)*i/400;return[x,fn(x)];}),values=points.map(point=>point[1]).filter(Number.isFinite).sort((a,b)=>a-b);
 if(!values.length)throw new Error('No real values in this graph range');let low=values[Math.floor(values.length*.02)],high=values[Math.floor((values.length-1)*.98)];if(low===high){low-=1;high+=1;}const padding=(high-low)*.12;
 const yMin=Number(spec.yMin??low-padding),yMax=Number(spec.yMax??high+padding);if(!Number.isFinite(yMin)||!Number.isFinite(yMax)||yMin>=yMax||Math.max(Math.abs(yMin),Math.abs(yMax))>1e12)throw new Error('Invalid graph vertical range');return{points,xMin,xMax,yMin,yMax,title:String(spec.title||'Graph of '+spec.expression).slice(0,160),expression:spec.expression};
}
const escape=text=>String(text).replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
export function graphSvg(spec){
 const graph=sampleGraph(spec),{xMin,xMax,yMin,yMax}=graph,w=620,h=390,p={left:66,right:22,top:42,bottom:52},width=w-p.left-p.right,height=h-p.top-p.bottom;
 const x=value=>p.left+(value-xMin)/(xMax-xMin)*width,y=value=>p.top+(yMax-value)/(yMax-yMin)*height;
 const fmt=value=>new Intl.NumberFormat('en',{maximumSignificantDigits:3}).format(value);let grid='',labels='';
 for(let i=0;i<=6;i++){const xv=xMin+(xMax-xMin)*i/6,yv=yMin+(yMax-yMin)*i/6;grid+=`<path d="M${x(xv)} ${p.top}V${p.top+height}M${p.left} ${y(yv)}H${p.left+width}"/>`;labels+=`<text x="${x(xv)}" y="${h-25}" text-anchor="middle">${escape(fmt(xv))}</text><text x="${p.left-9}" y="${y(yv)+4}" text-anchor="end">${escape(fmt(yv))}</text>`;}
 let path='',last=null;for(const point of graph.points){const px=x(point[0]),py=y(point[1]);if(!Number.isFinite(py)||py<p.top-2||py>p.top+height+2){last=null;continue;}path+=(last&&Math.abs(py-last)<height*.8?'L':'M')+px.toFixed(2)+' '+py.toFixed(2);last=py;}
 const axes=(xMin<=0&&xMax>=0?`M${x(0)} ${p.top}V${p.top+height}`:'')+(yMin<=0&&yMax>=0?`M${p.left} ${y(0)}H${p.left+width}`:'');
 return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${escape(graph.title)}"><title>${escape(graph.title)}</title><desc>Function ${escape(graph.expression)} from x=${xMin} to ${xMax}.</desc><rect width="100%" height="100%" fill="#fff"/><text x="${p.left}" y="25" fill="#253648" font-family="sans-serif" font-size="16">${escape(graph.title)}</text><g fill="none" stroke="#e4eaf0" stroke-width="1">${grid}</g><g fill="#526477" font-family="sans-serif" font-size="12">${labels}<text x="${w-12}" y="${h-25}">x</text><text x="30" y="${p.top-7}">y</text></g><path d="${axes}" stroke="#748399" fill="none" stroke-width="1.5"/><path d="${path}" stroke="#286a9b" fill="none" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
const graphBlocks=/```classio-graph\s*\n([\s\S]*?)\n```/g;
export function validateGraphBlocks(text){const blocks=[...text.matchAll(graphBlocks)];if(blocks.length!==(text.match(/```classio-graph/g)||[]).length||blocks.length>8)throw new Error('The graph specification is incomplete. Retrying this section.');for(const block of blocks)sampleGraph(JSON.parse(block[1]));return text;}
export function includeSourceGraph(text,source){
 if(text.includes('```classio-graph')||!/\b(?:graph|sketch|plot|end behavior)\b/i.test(source))return text;
 const normalized=source.replaceAll('−','-').replaceAll('²','^2').replaceAll('³','^3');
 const equation=normalized.match(/\b(?:[fg]\s*\(\s*x\s*\)|y)\s*=\s*([^;\n]+)/i)?.[1];if(!equation)return text;
 const expression=equation.trim().replace(/[.,]$/,'').replace(/(\d|\))(?=x|\()/g,'$1*').replace(/x(?=\()/g,'x*');
 try{sampleGraph({expression});return text+'\n\n```classio-graph\n'+JSON.stringify({expression,xMin:-5,xMax:5,title:'Graph of '+expression})+'\n```';}catch{return text;}
}
export async function notesWithMathGraphs(text,signal,generatedImages=new Set()){
 let result='',start=0;for(const match of text.matchAll(graphBlocks)){
  if(signal?.aborted)throw new DOMException('Cancelled','AbortError');const spec=JSON.parse(match[1]),svg=graphSvg(spec),url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'}));
  try{
   const image=new Image();await new Promise((resolve,reject)=>{const cancel=()=>reject(new DOMException('Cancelled','AbortError'));image.onload=()=>{signal?.removeEventListener('abort',cancel);resolve();};image.onerror=()=>{signal?.removeEventListener('abort',cancel);reject(new Error('The graph could not be drawn.'));};signal?.addEventListener('abort',cancel,{once:true});image.src=url;});
   if(signal?.aborted)throw new DOMException('Cancelled','AbortError');const canvas=document.createElement('canvas');canvas.width=1240;canvas.height=780;canvas.getContext('2d').drawImage(image,0,0,1240,780);
   const png=canvas.toDataURL('image/png');generatedImages.add(png);
   const label=String(spec.title||'Graph of '+spec.expression).replace(/[[\]\n]/g,' ');result+=text.slice(start,match.index)+`![${label}](${png})`;start=match.index+match[0].length;
  }finally{URL.revokeObjectURL(url);}
 }return result+text.slice(start);
}
