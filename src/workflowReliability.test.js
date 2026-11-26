import {synthesizeSpeech} from './podcastAudio';
import {likelyStudyImage} from './documentMedia';
import {parseAiJson} from './aiJson';
import {hasBlankContent,languageInstruction,studySource} from './studyContext';
import {normalizeChats,updateChat,persistentChats,newChat} from './chatModel';
import {parseLessonPlan} from './LearnMode';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));

afterEach(()=>localStorage.clear());
test('model JSON repairs single math backslashes and raw line breaks while preserving correctly escaped content',()=>{
  const broken=String.raw`[{"lesson":"\(F=ma\), \frac{v}{t} and \text{force}","title":"القوة"}]`;
  expect(parseAiJson(broken)[0]).toEqual({lesson:String.raw`\(F=ma\), \frac{v}{t} and \text{force}`,title:'القوة'});
  const valid={lesson:'First line\nSecond line \\frac{1}{2}',title:'مرحبا'};
  expect(parseAiJson(JSON.stringify(valid))).toEqual(valid);
  expect(parseAiJson('{"text":"First\nSecond"}').text).toBe('First\nSecond');
  expect(()=>parseAiJson('[{"lesson":')).toThrow('Your current work is kept');
  expect(()=>parseAiJson('window.alert("unsafe")')).toThrow();
});
test('original Arabic source takes priority over saved notes and determines output language',async()=>{
  const source=await studySource({notes:'An old English summary.',name:'Physics'},async()=> 'تتناول هذه الوحدة عملية البناء الضوئي في النباتات ودور الضوء في إنتاج الغذاء.');
  expect(source).toContain('البناء الضوئي');expect(source).toContain('An old English summary.');
  expect(languageInstruction(source)).toContain('in Arabic');
  localStorage.setItem('classio_lang','French');expect(languageInstruction(source)).toContain('in French');expect(languageInstruction(source,'Arabic')).toContain('in Arabic');
});
test('only empty blank notes are disposable; uploads and any learning content are kept',()=>{
  expect(hasBlankContent({id:'blank_1',_isBlank:true,notes:'',notesHtml:'<p><br></p>'})).toBe(false);
  for(const file of [{id:'upload',notes:''},{id:'blank_1',_fileObj:new Blob([''])},{id:'blank_1',quiz:{questions:[{}]}},{id:'blank_1',course:{units:[{}]}},{id:'blank_1',notesHtml:'<img src="x">'},{id:'blank_1',notes:'تعريف'},{id:'blank_1',generationJobs:{notes:{status:'waiting'}}},{id:'blank_1',pendingPresentation:{slides:[{}]}}])expect(hasBlankContent(file)).toBe(true);
});
test('legacy chats migrate and changing one conversation preserves the others',()=>{
  const record=normalizeChats([{role:'user',content:'Original question'}]),second=newChat();
  const multiple={...record,chats:[...record.chats,second],activeId:second.id};
  const changed=updateChat(multiple,second.id,{messages:[{role:'user',content:'Second question',image:'private-attachment'}],draft:'Keep this draft'});
  expect(changed.chats[0].messages[0].content).toBe('Original question');expect(changed.chats[1].draft).toBe('Keep this draft');
  expect(persistentChats(changed).chats[1].messages[0]).toEqual({role:'user',content:'Second question'});
  expect(normalizeChats(changed).activeId).toBe(second.id);
});
test('longer guided courses keep all their checks rather than rejecting after six units',()=>{
  const unit={title:'موضوع',lesson:'شرح',question:'سؤال',options:['أ','ب','ج'],correct:1,explanation:'تفسير'};
  expect(parseLessonPlan(JSON.stringify(Array.from({length:10},()=>unit)))).toHaveLength(10);
});

test('automatic illustrations reject a matching landscape photograph from Geograph',()=>{
 const forest={title:"Bomb Crater in King's Wood",context:"A photograph of a woodland crater",source:"https://commons.wikimedia.org/forest",src:"https://example.com/forest.jpg"};
 const diagram={title:"Bomb crater diagram",context:"A cross section of impact geometry",source:"https://commons.wikimedia.org/diagram",src:"https://example.com/diagram.png",kind:"diagram"};
 expect(likelyStudyImage('bomb crater',[forest])).toBeNull();expect(likelyStudyImage('bomb crater',[forest,diagram])).toBe(diagram);
});

test('speech waits for the actual capacity reset and retries the same turn automatically',async()=>{
 jest.useFakeTimers();const previous=global.fetch,controller=new AbortController(),states=[];const blob=new Blob(['fixture audio']);
 global.fetch=jest.fn().mockResolvedValueOnce({status:429,ok:false,headers:{get:()=>"45"}}).mockResolvedValueOnce({status:200,ok:true,headers:{get:()=>"audio/wav"},blob:async()=>blob});
 const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
 try{const pending=synthesizeSpeech("A useful study explanation.","nova",controller.signal,state=>states.push(state));await flush();expect(states).toEqual([true]);jest.advanceTimersByTime(20000);await flush();jest.advanceTimersByTime(20000);await flush();jest.advanceTimersByTime(4999);await flush();expect(global.fetch).toHaveBeenCalledTimes(1);jest.advanceTimersByTime(1);await flush();await expect(pending).resolves.toBe(blob);expect(global.fetch).toHaveBeenCalledTimes(2);expect(states).toEqual([true,false]);}finally{controller.abort();global.fetch=previous;jest.useRealTimers();}
});

test('a temporary voice outage retries the unfinished turn once and preserves cancellation',async()=>{
 jest.useFakeTimers();const previous=global.fetch,controller=new AbortController(),blob=new Blob(['audio']);global.fetch=jest.fn().mockResolvedValueOnce({status:503,ok:false,headers:{get:()=>null}}).mockResolvedValueOnce({status:200,ok:true,headers:{get:()=> 'audio/wav'},blob:async()=>blob});
 const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
 try{const pending=synthesizeSpeech('A clear explanation.','aria',controller.signal);await flush();jest.advanceTimersByTime(1999);await flush();expect(global.fetch).toHaveBeenCalledTimes(1);jest.advanceTimersByTime(1);await flush();await expect(pending).resolves.toBe(blob);expect(global.fetch).toHaveBeenCalledTimes(2);expect(global.fetch.mock.calls[0][1].body).toBe(global.fetch.mock.calls[1][1].body);}finally{controller.abort();global.fetch=previous;jest.useRealTimers();}
});
test('a persistent voice outage stops after a bounded retry and cancellation prevents a new attempt',async()=>{
 jest.useFakeTimers();const previous=global.fetch,controller=new AbortController();global.fetch=jest.fn().mockResolvedValue({status:503,ok:false,headers:{get:()=>null}});const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
 try{const pending=synthesizeSpeech('A clear explanation.','aria',controller.signal);const failed=expect(pending).rejects.toThrow('transcript is still available');await flush();jest.advanceTimersByTime(2000);await flush();await failed;expect(global.fetch).toHaveBeenCalledTimes(2);const cancelled=synthesizeSpeech('Another explanation.','aria',controller.signal);const stopped=expect(cancelled).rejects.toHaveProperty('name','AbortError');await flush();controller.abort();await stopped;jest.advanceTimersByTime(3000);await flush();expect(global.fetch).toHaveBeenCalledTimes(3);}finally{controller.abort();global.fetch=previous;jest.useRealTimers();}
});
