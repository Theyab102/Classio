import katex from 'katex';
import {normalizeMathSource} from './mathSyntax';
import {normalizeLatex} from './mathMarkdown';
const operators={'=':'equals','+':'plus','−':'minus','-':'minus','×':'times','⋅':'times','/':'divided by','±':'plus or minus','≈':'approximately equals','→':'yields','θ':'theta','α':'alpha','β':'beta','π':'pi','∑':'sum','∫':'integral','∞':'infinity'};
function speechMath(latex){
  latex=normalizeLatex(latex);
  const simple=latex.replace(/[\s{}\\]/g,'');
  if(/^F=ma$/.test(simple))return 'Force equals mass multiplied by acceleration';
  if(/^v=u\+at$/.test(simple))return 'Final velocity equals initial velocity plus acceleration multiplied by time';
  try{
    const html=katex.renderToString(latex,{output:'mathml',throwOnError:true,trust:false,maxExpand:500}),document=new DOMParser().parseFromString(html,'text/html');
    function read(node){if(node.nodeType===3)return node.textContent;if(!node.tagName)return '';const name=node.tagName.toLowerCase(),children=[...node.children];if(name==='annotation')return '';if(name==='semantics')return read(children[0]);if(name==='mfrac')return '('+read(children[0])+') over ('+read(children[1])+')';if(name==='msqrt')return 'square root of '+children.map(read).join(' ');if(name==='mroot')return read(children[1])+' root of '+read(children[0]);if(name==='msup'){const power=read(children[1]);return read(children[0])+(power==='2'?' squared':power==='3'?' cubed':' to the power of '+power);}if(name==='msub')return read(children[0])+' subscript '+read(children[1]);if(name==='msubsup')return read(children[0])+' subscript '+read(children[1])+' to the power of '+read(children[2]);if(['mo','mi'].includes(name))return operators[node.textContent]||node.textContent;if(!children.length)return node.textContent;return children.map(read).join(' ');}
    return read(document.querySelector('math')).replace(/\s+/g,' ').trim();
  }catch{return 'Please review the equation in the transcript';}
}
export function spokenMathText(text){
  return normalizeMathSource(String(text||'')).replace(/\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$|\$([^$\n]+)\$/g,(_,a,b,c,d)=>speechMath(a||b||c||d))
    .replace(/\\(?:t?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g,'$1 over $2').replace(/\\sqrt\s*\{([^{}]+)\}/g,'square root of $1')
    .replace(/\\(?:text|mathrm)\s*\{([^{}]+)\}/g,'$1').replace(/\\(?:times|cdot)\b/g,' times ').replace(/\\approx\b/g,' approximately ').replace(/\\[A-Za-z]+\b/g,' ').replace(/\\[()[\],;!]/g,' ')
    .replace(/(?<![A-Za-z])m\/s(?:\^\{?2\}?|²)/g,'metres per second squared').replace(/(?<![A-Za-z])m\/s/g,'metres per second').replace(/√/g,' square root of ').replace(/≈/g,' approximately ').replace(/²/g,' squared ').replace(/³/g,' cubed ').replace(/θ/g,'theta').replace(/[{}]/g,'').replace(/\*\*|__|(?<=^|\n)#{1,6}\s*/g,'').replace(/\s+/g,' ').trim();
}
