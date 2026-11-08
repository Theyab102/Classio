import {normalizeMathSource} from './mathSyntax';
const formula=/\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\$[^$\n]+\$/g;
// Preview length is a soft bound: finish an equation rather than cutting its delimiters.
export function studyExcerpt(value,limit=125){
 const equations=[];
 const protectedText=normalizeMathSource(String(value||'')).replace(formula,expression=>{equations.push(expression);return`CLASSIOPREVIEWMATH${equations.length-1}END`;});
 const text=protectedText.replace(/<[^>]*>/g,' ').replace(/!?\[([^\]]*)\]\([^)]+\)/g,'$1').replace(/&nbsp;/gi,' ').replace(/&amp;/gi,'&').replace(/(?:^|\n)#{1,6}\s*/g,'').replace(/\s+/g,' ').trim().replace(/CLASSIOPREVIEWMATH(\d+)END/g,(_,index)=>equations[Number(index)]);
 let end=Math.min(limit,text.length),match;formula.lastIndex=0;
 while((match=formula.exec(text)))if(match.index<end&&formula.lastIndex>end)end=formula.lastIndex;
 return text.slice(0,end)+(end<text.length?'…':'');
}
