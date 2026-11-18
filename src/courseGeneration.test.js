import {courseSections,generateCourseLessons,parseLessonPlan} from './courseGeneration';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
const lesson=(title='Lesson')=>({title,lesson:'## Explanation\n\nA complete worked example and takeaway.',question:'What follows?',options:['One','Two','Three'],correct:0,explanation:'One follows from the example.'});
const outline=count=>JSON.stringify(Array.from({length:count},(_,i)=>({title:`Section ${i+1}`,focus:`Explain section ${i+1}`})));

test('partitions all source text in order, including the final section and Unicode',()=>{
  const text=('مصدر عن الحركة 🌞 ومعادلاتها.\n\n').repeat(900)+'FINAL SECTION';
  const sections=courseSections(text,12);
  expect(sections.join('')).toBe(text);expect(sections[11]).toContain('FINAL SECTION');
  expect(sections.every(part=>part.length>0&&!/^[\uDC00-\uDFFF]/.test(part))).toBe(true);
});
test('a 12-lesson course uses one outline and separate bounded lesson requests, with later source coverage',async()=>{
  const source=Array.from({length:12},(_,i)=>`SOURCE_${i} ${'evidence '.repeat(100)}\n\n`).join('');
  const send=jest.fn().mockResolvedValueOnce(outline(12));
  for(let i=0;i<12;i++)send.mockResolvedValueOnce(JSON.stringify([lesson('Lesson '+i)]));
  const progress=[];const result=await generateCourseLessons({source,topic:'Test',count:12,send,onProgress:message=>progress.push(message)});
  expect(result).toHaveLength(12);expect(send).toHaveBeenCalledTimes(13);
  expect(send.mock.calls[12][0].messages[1].content).toContain('SOURCE_11');
  expect(send.mock.calls[1][0].messages[1].content).not.toContain('SOURCE_11');
  expect(send.mock.calls.slice(1).every(([request])=>request.maxTokens<=2600)).toBe(true);
  expect(progress.at(-1)).toBe('Writing lesson 12 of 12…');
});
test('retries a provider-truncated lesson with a shorter prompt without regenerating prior lessons',async()=>{
  const send=jest.fn().mockResolvedValueOnce(outline(3)).mockResolvedValueOnce(JSON.stringify([lesson('First')]))
    .mockRejectedValueOnce(Object.assign(new Error('Too long'),{code:'OUTPUT_LIMIT'})).mockResolvedValueOnce(JSON.stringify([lesson('Second')])).mockResolvedValueOnce(JSON.stringify([lesson('Third')]));
  const units=await generateCourseLessons({source:'Source evidence '.repeat(120),topic:'Test',count:3,send});
  expect(units.map(unit=>unit.title)).toEqual(['First','Second','Third']);expect(send).toHaveBeenCalledTimes(5);
  expect(send.mock.calls[3][0].maxTokens).toBe(1400);expect(send.mock.calls[3][0].messages[0].content).toContain('under 900 characters');
});
test('malformed/truncated JSON gets one bounded repair, and retries never exceed 15 for 12 lessons',async()=>{
  const send=jest.fn().mockResolvedValueOnce(outline(12));
  send.mockResolvedValueOnce('[{"title":"broken","lesson":"'+('realistic incomplete output '.repeat(160)))
    .mockResolvedValueOnce(JSON.stringify([lesson("Lesson first")]))
    .mockRejectedValueOnce(Object.assign(new Error('Too long'),{code:'OUTPUT_LIMIT'})).mockResolvedValueOnce(JSON.stringify([lesson("Lesson second")]));
  for(let i=2;i<11;i++)send.mockResolvedValueOnce(JSON.stringify([lesson("Lesson "+i)]));
  send.mockRejectedValueOnce(Object.assign(new Error('Too long'),{code:'OUTPUT_LIMIT'}));
  await expect(generateCourseLessons({source:'Evidence '.repeat(200),topic:'Test',count:12,send})).rejects.toThrow('Too long');
  expect(send).toHaveBeenCalledTimes(15);
});
test('a rate limit stops immediately; cancellation ignores a late provider response',async()=>{
  const send=jest.fn().mockRejectedValue(Object.assign(new Error('Wait 45 seconds'),{code:'RATE_LIMIT',status:429}));
  await expect(generateCourseLessons({source:'Text',count:3,send})).rejects.toThrow('45 seconds');expect(send).toHaveBeenCalledTimes(1);
  const controller=new AbortController();let resolve;
  const late=jest.fn(()=>new Promise(done=>{resolve=done;}));
  const job=generateCourseLessons({source:'Text',count:3,signal:controller.signal,send:late});controller.abort();resolve(outline(3));
  await expect(job).rejects.toMatchObject({name:'AbortError'});expect(late).toHaveBeenCalledTimes(1);
});
test('does not silently clip a lesson or manufacture missing knowledge checks',()=>{
  expect(()=>parseLessonPlan(JSON.stringify([{...lesson(),lesson:'x'.repeat(10001)}]))).toThrow('oversized');
  expect(()=>parseLessonPlan(JSON.stringify([{...lesson(),options:[]}]))).toThrow('incomplete');
});
test('short sources stay intact in each focused lesson instead of becoming empty fragments',async()=>{
 const source='تشرح هذه الفقرة الحركة والقوة وعلاقتهما بالتسارع.';
 const send=jest.fn().mockResolvedValueOnce(outline(3));for(let i=0;i<3;i++)send.mockResolvedValueOnce(JSON.stringify([lesson('Lesson '+i)]));
 await generateCourseLessons({source,topic:'A filename is not evidence',count:3,send});
 for(const [request] of send.mock.calls)expect(request.messages[1].content).toContain(source);
});
