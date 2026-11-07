const greek={theta:'\\theta',alpha:'\\alpha',beta:'\\beta',gamma:'\\gamma',pi:'\\pi',Delta:'\\Delta',omega:'\\omega'};
// A small expression grammar, not a rewrite of arbitrary prose. Parentheses
// explicitly delimit compound numerators, denominators and function arguments.
export function plainMathToLatex(input){
  let source=String(input).trim();if(!source||source.length>500||/[\\<>@:#]/.test(source))return null;
  if(/^\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}$/.test(source)||/\/\s*(?:\d+[A-Za-z]|[A-Za-z]{2,})\b/.test(source))return null;
  if(/^(?:[A-Z][a-z]?(?:_\{?\d+\}?)?){2,}$/.test(source)&&/_/.test(source))return '\\mathrm{'+source.replace(/_(\d+)/g,'_{$1}')+'}';
  source=source.replace(/\+\/-|\+-/g,'±').replace(/×/g,'*').replace(/²/g,'^2').replace(/³/g,'^3').replace(/√/g,'sqrt');
  const tokens=[];let index=0;
  while(index<source.length){if(/\s/.test(source[index])){index++;continue;}const rest=source.slice(index),match=/^(?:\d+(?:\.\d+)?|[A-Za-z]+|[()+\-*/^_=±{}])/.exec(rest);if(!match)return null;tokens.push(match[0]);index+=match[0].length;}
  if(!tokens.length||!tokens.some(t=>['=','+','-','*','/','^','_','±','sqrt'].includes(t)))return null;
  let at=0;
  const group=()=>{const open=tokens[at++],close=open==='(' ? ')':'}';const value=expression(0);if(tokens[at++]!==close)throw Error('Unbalanced expression');return value;};
  function atom(){const token=tokens[at++];if(token===undefined)throw Error('Missing operand');if(token==='('||token==='{'){at--;return '\\left('+group()+'\\right)';}if(token==='-'||token==='+')return token+expression(35);if(token==='sqrt'){if(!['(','{'].includes(tokens[at]))throw Error('Root needs parentheses');return '\\sqrt{'+group()+'}';}if(['sin','cos','tan','log','ln'].includes(token)){if(!['(','{'].includes(tokens[at]))throw Error('Function needs parentheses');return '\\'+token+'\\left('+group()+'\\right)';}if(greek[token])return greek[token];if(/^\d/.test(token))return token;if(/^[A-Za-z]{1,2}$/.test(token))return token;throw Error('Unrecognized word');}
  function expression(min){let left=atom();for(;;){const token=tokens[at];if(token===undefined||token===')'||token==='}')break;
      if(token==='_'){at++;let sub;if(tokens[at]==='{'){at++;const parts=[];while(at<tokens.length&&tokens[at]!=='}')parts.push(tokens[at++]);if(tokens[at++]!=='}'||!parts.length)throw Error('Bad subscript');sub=parts.join('');}else{sub=tokens[at++];if(!/^[a-zA-Z0-9]+$/.test(sub||''))throw Error('Bad subscript');if(/^\d+$/.test(sub)&&/^[a-zA-Z]{1,2}$/.test(tokens[at]||''))sub+=tokens[at++];}left+='_{'+sub+'}';continue;}
      const implicit=/^(?:\d|[A-Za-z(])/.test(token),precedence=token==='='?5:['+','-','±'].includes(token)?10:token==='*'||token==='/'||implicit?20:token==='^'?40:-1;
      if(precedence<min||precedence<0)break;if(!implicit)at++;const right=expression(token==='^'?precedence:precedence+1);
      if(token==='/')left='\\frac{'+ungroup(left)+'}{'+ungroup(right)+'}';else if(token==='^')left+='^{'+ungroup(right)+'}';else if(implicit)left+=right;else if(token==='*')left+=(/^[a-zA-Z]$/.test(left)&&/^[a-zA-Z]$/.test(right)?'':'\\cdot ')+right;else left+=(token==='±'?'\\pm ':token)+right;
    }return left;}
  function ungroup(value){return value.startsWith('\\left(')&&value.endsWith('\\right)')?value.slice(6,-7):value;}
  try{const result=expression(0);return at===tokens.length?result:null;}catch{return null;}
}
export function normalizeMathSource(input){
  const protectedParts=[];
  let text=String(input||'').replace(/```[\s\S]*?```|`[^`\n]+`|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|https?:\/\/\S+/g,value=>{const key='CLASSIOPROTECTED'+protectedParts.length+'TOKEN';protectedParts.push(value);return key;});
  const formula=value=>{const latex=/[A-Za-z_=^±]/.test(value)?plainMathToLatex(value):null;return latex?'\\('+latex+'\\)':value;};
  text=text.split('\n').map(line=>{if(/CLASSIOPROTECTED/.test(line))return line;if(/^\s*\|/.test(line))return line.split('|').map(cell=>formula(cell)).join('|');const match=/^(\s*(?:(?:#{1,6}|>|[-*•]|\d+[.)])\s+)+)(.*)$/.exec(line);return match?match[1]+formula(match[2]):formula(line);}).join('\n');
  text=text.replace(/\b([A-Za-z](?:_\{?[a-zA-Z0-9]+\}?)?\s*=\s*(?:sqrt\([^)]*\)|[A-Za-z0-9_{}^+*/()±-]|[ \t]){1,120}?)(?=\s+[A-Za-z]{3,}\b|[,;.!?]|$)/gm,(value)=>{const latex=plainMathToLatex(value);return latex?'\\('+latex+'\\)':value;});
  return text.replace(/CLASSIOPROTECTED(\d+)TOKEN/g,(_,index)=>protectedParts[Number(index)]);
}
