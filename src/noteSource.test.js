import { resolveNoteSource } from './noteSource';
jest.mock('./studyTools',()=>({TTS_SPACE_URL:'https://tts.example.test'}));
let originalFetch;
beforeEach(()=>{originalFetch=global.fetch;});
afterEach(()=>{global.fetch=originalFetch;});

test('pasted source and uploaded files take precedence over existing notes',async()=>{
  const getFileText=jest.fn(async()=> 'Actual source content from the uploaded file.');
  await expect(resolveNoteSource({file:{},pastedText:'  User supplied transcript.  ',existing:'My writing.',getFileText})).resolves.toEqual({text:'User supplied transcript.',label:'supplied source text'});
  expect(getFileText).not.toHaveBeenCalled();
  await expect(resolveNoteSource({file:{},existing:'My writing.',getFileText})).resolves.toMatchObject({text:'Actual source content from the uploaded file.'});
});

test('saved source transcript can be reused without refetching the link',async()=>{
  global.fetch=jest.fn();
  await expect(resolveNoteSource({file:{youtubeUrl:'https://youtu.be/dQw4w9WgXcQ',transcript:'An available saved source transcript.'}})).resolves.toMatchObject({text:'An available saved source transcript.'});
  expect(global.fetch).not.toHaveBeenCalled();
});

test('missing video transcripts reject with a paste option rather than treating the title as analyzed content',async()=>{
  global.fetch=jest.fn(async()=>({ok:true,json:async()=>({success:false})}));
  await expect(resolveNoteSource({file:{name:'Video title',youtubeUrl:'https://youtu.be/dQw4w9WgXcQ'},existing:'Some earlier writing.'})).rejects.toThrow('video has not been analyzed');
  expect(global.fetch.mock.calls[0][0]).toBe('https://tts.example.test/transcript/dQw4w9WgXcQ');
});

test('available captions return their real text and unsafe links never initiate requests',async()=>{
  const transcript='This is the actual transcript that was made available by the caption service.';
  global.fetch=jest.fn(async()=>({ok:true,json:async()=>({success:true,transcript})}));
  await expect(resolveNoteSource({file:{youtubeUrl:'https://youtube.com/shorts/dQw4w9WgXcQ'}})).resolves.toMatchObject({text:transcript});
  global.fetch.mockClear();
  await expect(resolveNoteSource({file:{sourceUrl:'javascript:alert(1)'}})).rejects.toThrow();
  expect(global.fetch).not.toHaveBeenCalled();
});

test('readable websites strip executable and navigation content; inaccessible websites fail honestly',async()=>{
  global.fetch=jest.fn(async()=>({ok:true,text:async()=>'<header>Navigation</header><script>doBadThings()</script><main>'+ 'Study content. '.repeat(15)+'</main>'}));
  const result=await resolveNoteSource({file:{sourceUrl:'https://example.com/lesson'}});
  expect(result.text).toContain('Study content.');
  expect(result.text).not.toMatch(/Navigation|doBadThings/);
  global.fetch=jest.fn(async()=>{throw new Error('CORS');});
  await expect(resolveNoteSource({file:{sourceUrl:'https://example.com/lesson'}})).rejects.toThrow('Paste the source text');
});

 test('image filename hints are never used as invented source facts',async()=>{
  await expect(resolveNoteSource({file:{},getFileText:async()=> '[This is an image file: biology.png. Describe its likely content based on the filename.]'})).rejects.toThrow('image has no extracted text');
});
