import { captionAt, parseDialogue, singleVoiceDialogue } from './podcastAudio';
import { sanitizeNoteHtml } from './noteFormat';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
jest.mock('./firebase',()=>({accountHeaders:jest.fn(async()=>({}))}));

test('captions follow media time across gaps, exact boundaries and arbitrary seeks',()=>{
 const cues=[{start:0,end:1.25,speaker:0},{start:1.5,end:4.2,speaker:1}];
 expect(captionAt(cues,0)).toBe(0);expect(captionAt(cues,1.25)).toBe(-1);expect(captionAt(cues,2.9)).toBe(1);expect(captionAt(cues,.4)).toBe(0);expect(captionAt(cues,5)).toBe(-1);
});
test('AI dialogue requires actual valid turns for both hosts and rejects malformed success content',()=>{
 expect(parseDialogue(JSON.stringify([{speaker:0,text:'Explain energy.'},{speaker:1,text:'How does it transfer?'}]))).toHaveLength(2);
 expect(()=>parseDialogue('A fabricated non-JSON answer.')).toThrow('unreadable');
 expect(()=>parseDialogue('[{"speaker":0,"text":"Only one host."}]')).toThrow('incomplete');
 expect(()=>parseDialogue('[{"speaker":9,"text":"Invalid host."}]')).toThrow('incomplete');
});
test('legacy scripts split without losing text and never silently truncate oversized sentences',()=>{
 const script='Energy changes form. '.repeat(90);
 const turns=singleVoiceDialogue(script);expect(turns.length).toBeGreaterThan(1);expect(turns.map(turn=>turn.text).join(' ')).toBe(script.trim());
 expect(()=>singleVoiceDialogue('x'.repeat(1201))).toThrow('shorter sentences');
});
test('note font settings survive save/reload sanitization while remote CSS and executable styles are stripped',()=>{
 const result=sanitizeNoteHtml('<p><span style="font-family:Fraunces Variable;font-size:24px;background-image:url(https://tracker.invalid/a)">Title</span><span style="font-family:remote-font;font-size:999px">Other</span></p><style>@import url(https://fonts.googleapis.com/css);</style>');
 expect(result).toContain('font-size: 24px');expect(result).toContain('Fraunces Variable');expect(result).not.toMatch(/tracker.invalid|googleapis|remote-font|999px|background-image/);
});
