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
  for(const file of [{id:'upload',notes:''},{id:'blank_1',_fileObj:new Blob([''])},{id:'blank_1',quiz:{questions:[{}]}},{id:'blank_1',course:{units:[{}]}},{id:'blank_1',notesHtml:'<img src="x">'},{id:'blank_1',notes:'تعريف'}])expect(hasBlankContent(file)).toBe(true);
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

test('speech rate limits show a retry interval instead of a generic generation error',async()=>{const previous=global.fetch;global.fetch=jest.fn(async()=>({status:429,ok:false,headers:{get:()=>"45"}}));try{await expect(synthesizeSpeech("A useful study explanation.","nova",new AbortController().signal)).rejects.toThrow("45 seconds");}finally{global.fetch=previous;}});
