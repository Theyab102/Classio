import {render} from '@testing-library/react';
import StudyMarkdown from './StudyMarkdown';
import {normalizeMathSource,plainMathToLatex} from './mathSyntax';
import {markdownToNoteHtml} from './noteFormat';
import {typesetMath} from './mathMarkdown';
import {studyExcerpt} from './studyExcerpt';
jest.mock('./pdf',()=>({loadPdfLibrary:jest.fn(),openPdf:jest.fn()}));

const outsideMath = root => {const walker=document.createTreeWalker(root,NodeFilter.SHOW_TEXT),text=[];while(walker.nextNode())if(!walker.currentNode.parentElement.closest('[data-classio-equation],code,pre'))text.push(walker.currentNode.textContent);return text.join(' ');};
test.each([false,true])('the shared renderer covers physics, calculus, matrices and chemistry in block/inline surfaces (%s)',inline=>{
 const text=String.raw`Vertical: \(a_y=-g\); \(v_y=v_{0y}-gt\). A fraction \(\frac{1}{2}mv^2\), a root \(\sqrt{x+1}\), and \(\int_0^1 x^2\,dx=\frac{1}{3}\).

\[\begin{aligned}v_x&=v_0\cos\theta\\v_y&=v_0\sin\theta-gt\end{aligned}\]

Matrix \(\begin{pmatrix}a&b\\c&d\end{pmatrix}\). Chemistry \(\ce{2H2 + O2 -> 2H2O}\).`;
 const {container}=render(<StudyMarkdown text={text} inline={inline}/>);
 expect(container.querySelectorAll('.katex')).toHaveLength(8);expect(container.querySelector('.math-needs-review')).toBeNull();expect(container.querySelector('mfrac')).not.toBeNull();expect(container.querySelector('mtable')).not.toBeNull();expect(outsideMath(container)).not.toMatch(/\\[()[\]]|\\(?:frac|sqrt|ce|begin)/);
});
test('the exact missed vertical-velocity choices use structured subscripts and preserve their prose',()=>{
 const text='It decreases on ascent, following vy = v0y - g t. It is zero at the apex.';
 const normalized=normalizeMathSource(text);expect(normalized).toContain('v_{y}');expect(normalized).toContain('v_{0y}');const {container}=render(<StudyMarkdown text={text}/>);expect(container.querySelector('.katex')).not.toBeNull();expect(outsideMath(container)).toContain('It is zero at the apex.');
});
test('double-escaped equations from stored model responses recover without altering code examples',()=>{
 const text=String.raw`Use \\(v_{y}=v_{0y}-gt\\), then \\[x=\\frac{1}{2}gt^2\\].`+'\n\n`\\\\(literal code\\\\)`';
 const host=document.createElement('div');host.innerHTML=markdownToNoteHtml(text);typesetMath(host);expect(host.querySelectorAll('.katex')).toHaveLength(2);expect(host.querySelector('.math-needs-review')).toBeNull();expect(host.querySelector('code').textContent).toContain('\\\\(');
});
test.each(['v0x = v0*cos(theta)','vy = v0y - g*t','θ ≤ π/2','x ≠ 0','F ≈ m*a'])('common keyboard notation remains renderable: %s',text=>{
 expect(plainMathToLatex(text)).toBeTruthy();const host=document.createElement('div');host.innerHTML=markdownToNoteHtml(text);typesetMath(host);expect(host.querySelector('.katex')).not.toBeNull();expect(host.querySelector('.math-needs-review')).toBeNull();
});

test('material previews finish fractions instead of slicing through equation delimiters',()=>{
 const text='A worked example. '+String.raw`\(v=\frac{a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v+w+x+y+z}{1+2+3+4+5+6+7+8+9+10}\)`+' More content here.';
 const preview=studyExcerpt(text,60),{container}=render(<StudyMarkdown inline text={preview}/>);expect(container.querySelector('mfrac')).not.toBeNull();expect(container.querySelector('.math-needs-review')).toBeNull();expect(outsideMath(container)).toContain('A worked example.');expect(studyExcerpt(String.raw`\(x < y > z\)`)).toContain('x < y > z');
});
