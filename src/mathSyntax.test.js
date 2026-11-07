import {plainMathToLatex,normalizeMathSource} from './mathSyntax';
import {spokenMathText} from './mathSpeech';
import {markdownToNoteHtml} from './noteFormat';
import {typesetMath} from './mathMarkdown';
import katex from 'katex';
jest.mock('./pdf',()=>({loadPdfLibrary:jest.fn(),openPdf:jest.fn()}));
test.each(['x^2 + 3x + 2 = 0','(-b +- sqrt(b^2-4ac))/(2a)','F = m*a','v = u + a*t','1/2*m*v^2','H_2O'])('supported keyboard expression is valid structured math: %s',source=>{const latex=plainMathToLatex(source);expect(latex).toBeTruthy();expect(()=>katex.renderToString(latex,{throwOnError:true})).not.toThrow();});
test('ordinary dialogue, dates, file names and ambiguous divisions stay as text',()=>{for(const text of ['2026/11/07','Chapter 2 was written in 2025.','Week_4_notes.pdf','a/2x'])expect(normalizeMathSource(text)).toBe(text);expect(normalizeMathSource('I have 12 notes.')).toBe('I have 12 notes.');});
test('inline math and stored TeX render safely and bad syntax requests review rather than showing commands',()=>{const host=document.createElement('div');host.innerHTML=markdownToNoteHtml('Use F = ma.\n\n\\(t=\\sqrt{2h/g}\\)\n\n\\(\\notacommand{x}\\)');typesetMath(host);expect(host.querySelectorAll('.katex').length).toBe(2);expect(host.querySelector('.katex-error')).toBeNull();expect(host.textContent).toContain('Equation needs review');});
test('speech reads equations and units as words while normal dialogue remains normal',()=>{expect(spokenMathText('Force: \\(F=ma\\).')).toContain('Force equals mass multiplied by acceleration');expect(spokenMathText('Distance is \\(\\frac{a}{b}\\).')).toContain('over');expect(spokenMathText('We have 12 examples.')).toBe('We have 12 examples.');expect(spokenMathText('Travel at 100 m/s.')).toContain('metres per second');});

test('plain equations in numbered lists and table cells keep their Markdown structure',()=>{expect(normalizeMathSource('3. (-b +- sqrt(b^2 - 4*a*c))/(2*a)')).toEqual(expect.stringContaining('3. \\('));expect(normalizeMathSource('| Equation | (-b +- sqrt(b^2 - 4*a*c))/(2*a) |')).toContain('\\frac');});
