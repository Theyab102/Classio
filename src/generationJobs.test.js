import 'fake-indexeddb/auto';
import {setAiIdentityProvider} from './aiAccess';
import {jobStorage,runCheckpointJob,readGenerationJob,deleteGenerationJobs,distinct,appliedGenerationPatch} from './generationJobs';
import {generatePresentation,generateNotes,generateStudySet} from './checkpointGenerators';
import {generateCourseLessons} from './courseGeneration';
import {parseQuiz,isQuizCorrect} from './QuizStudio';
import {parseFlashcards} from './FlashcardStudio';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
let owner;beforeEach(()=>{owner='fixture-'+expect.getState().currentTestName;setAiIdentityProvider(()=>({uid:owner}));});
const request={task:'presentation',messages:[{role:'system',content:'Create a JSON educational slide deck.'},{role:'user',content:'Teach forces.'}],maxTokens:5000};
const fail=()=>Object.assign(new Error('Temporary failure'),{status:503});

test('a 20-slide deck persists slides 1–14 and resumes at slide 15 without repeating its outline or saved slides',async()=>{
 let broken=true;const seen=[];const send=jest.fn(async payload=>{
  const system=payload.messages[0].content;
  if(system.startsWith('Plan exactly'))return JSON.stringify({title:'Forces',designs:[],outline:Array.from({length:20},(_,i)=>({title:'Focus '+(i+1),focus:'Explain distinct concept '+(i+1),type:'content'}))});
  const number=Number(/Write ONLY slide (\d+)/.exec(system)[1]);seen.push(number);if(number===15&&broken)throw fail();return JSON.stringify({title:'Forces',slides:[{type:'content',title:'Concept '+number,bullets:['Accurate explanation '+number]}]});
 });
 const options={fileId:'large-deck',input:{request,count:20,topic:'Forces',source:'Source evidence '.repeat(900)},send};
 await expect(generatePresentation(options)).rejects.toThrow('Temporary failure');
 const saved=await jobStorage.get(owner,'large-deck','presentation');expect(saved.sections).toHaveLength(14);expect(saved.status).toBe('failed');expect(saved.outline.outline).toHaveLength(20);
 broken=false;seen.length=0;const completed=await generatePresentation({...options,resume:saved,input:{count:1}});
 expect(seen).toEqual([15,16,17,18,19,20]);expect(completed.slides).toHaveLength(20);expect(new Set(completed.slides.map(slide=>slide.id)).size).toBe(20);expect(send.mock.calls.filter(([p])=>p.messages[0].content.startsWith('Plan exactly'))).toHaveLength(1);
 expect((await jobStorage.get(owner,'large-deck','presentation')).status).toBe('ready');
});
test('long notes save their outline and complete sections, retain math and resume only the failed section',async()=>{
 let broken=true,sections=0;const source='A source paragraph with mathematics $x^2$ and forces.\n\n'.repeat(400);
 const send=jest.fn(async payload=>{if(payload.messages[0].content.startsWith('Plan exactly'))return JSON.stringify(Array.from({length:3},(_,i)=>({title:'Section '+i,focus:'Focus '+i})));const index=Number(/ONLY section (\d+)/.exec(payload.messages[1].content)[1]);sections++;if(index===2&&broken)throw fail();return `## Section ${index}\n\nUnique section ${index}: $\\frac{1}{2}mv^2$.`;});
 const options={fileId:'notes',input:{source,topic:'Energy',request:{...request,task:'notes'}},send};
 await expect(generateNotes(options)).rejects.toThrow();const saved=await jobStorage.get(owner,'notes','notes');expect(saved.sections).toHaveLength(1);
 broken=false;sections=0;const notes=await generateNotes({...options,resume:saved});expect(sections).toBe(2);expect(notes).toContain('\\frac{1}{2}');expect(notes.match(/## Section 1/g)).toHaveLength(1);
});
test('courses save each validated lesson and resume without regenerating the outline or earlier knowledge checks',async()=>{
 let broken=true;const seen=[];const send=jest.fn(async payload=>{if(payload.messages[0].content.includes('compact course outline'))return JSON.stringify([1,2,3].map(i=>({title:'Lesson '+i,focus:'Focus '+i})));const i=Number(/Lesson (\d+) of/.exec(payload.messages[1].content)[1]);seen.push(i);if(i===3&&broken)throw fail();return JSON.stringify([{title:'Lesson '+i,lesson:'## Learning\n\nEvidence '+i,question:'Question '+i,options:['One','Two','Three'],correct:0,explanation:'Evidence '+i}]);});
 const options={fileId:'course',source:'Source evidence '.repeat(100),topic:'Forces',count:3,send};await expect(generateCourseLessons(options)).rejects.toThrow();const saved=await jobStorage.get(owner,'course','course');expect(saved.sections).toHaveLength(2);broken=false;seen.length=0;expect(await generateCourseLessons({...options,resume:saved})).toHaveLength(3);expect(seen).toEqual([3]);
});
test('quiz and flashcard batches persist exact validated counts and never append duplicate questions',async()=>{
 for(const kind of ['quiz','flashcards']){
  const size=kind==='quiz'?8:10;let broken=true;
  const send=jest.fn(async payload=>{const prior=payload.messages[1].content.includes('Question 0');if(prior&&broken)throw fail();const batch=Number(/Create exactly (\d+)/.exec(payload.messages[0].content)[1]);return JSON.stringify(Array.from({length:batch},(_,i)=>kind==='quiz'?{question:'Question '+(prior?i+size:i),options:['A','B','C','D'],correct:0,hint:'Reason',explanation:'Explanation'}:{question:'Question '+(prior?i+size:i),answer:'Answer '+i}));});
  const options={fileId:'set-'+kind,kind,input:{source:'Evidence',topic:'Study',count:size+3},parse:kind==='quiz'?parseQuiz:parseFlashcards,send};
  await expect(generateStudySet(options)).rejects.toThrow();const saved=await jobStorage.get(owner,options.fileId,kind);expect(saved.sections.flat()).toHaveLength(size);broken=false;const total=await generateStudySet({...options,resume:saved});expect(total).toHaveLength(size+3);expect(new Set(total.map(item=>item.id)).size).toBe(size+3);
 }
 expect(()=>distinct([{question:' SAME   Question '},{question:'same question'}],item=>item.question)).toThrow('repeated');
});
test('cancellation saves progress; reloading from durable storage resumes remaining steps without duplicates',async()=>{
 const controller=new AbortController(),seen=[];
 const options={fileId:'cancel',kind:'quiz',input:{count:4},total:4,step:async index=>{seen.push(index);return{question:'Question '+index};},onCheckpoint:job=>{if(job.sections.length===2)controller.abort();},signal:controller.signal};
 await expect(runCheckpointJob(options)).rejects.toMatchObject({name:'AbortError'});const saved=await readGenerationJob({id:'cancel'},'quiz');expect(saved.status).toBe('paused');expect(saved.sections).toHaveLength(2);
 const result=await runCheckpointJob({...options,signal:undefined,onCheckpoint:()=>{},resume:saved});expect(result).toHaveLength(4);expect(seen).toEqual([0,1,2,3]);await deleteGenerationJobs('cancel');expect(await readGenerationJob({id:'cancel'},'quiz')).toBeNull();
});
test('account switches cannot read or apply another student’s checkpoints or late output',async()=>{
 const alice=owner;let release;const options={fileId:'private',kind:'quiz',input:{},total:1,step:()=>new Promise(resolve=>{release=resolve;})};
 const pending=runCheckpointJob(options);while(!release)await new Promise(resolve=>setTimeout(resolve,0));const saved=await jobStorage.get(alice,'private','quiz');owner='fixture-bob';release('Private answer');await expect(pending).rejects.toMatchObject({name:'AbortError'});
 expect(await readGenerationJob({id:'private',generationJobs:{quiz:saved}},'quiz')).toBeNull();expect(await jobStorage.get(owner,'private','quiz')).toBeUndefined();
 await expect(runCheckpointJob({...options,resume:saved})).rejects.toThrow('does not belong');
});
test('storage failure stops generation before spending provider quota; concurrent tabs cannot claim one job',async()=>{
 const step=jest.fn(),store={put:async()=>{throw Error('Storage full');}};await expect(runCheckpointJob({fileId:'full',kind:'quiz',input:{},total:1,step,store})).rejects.toThrow('Storage full');expect(step).not.toHaveBeenCalled();
 await jobStorage.claim(owner,'locked','quiz','first');await expect(jobStorage.claim(owner,'locked','quiz','second')).rejects.toThrow('another tab');await jobStorage.release(owner,'locked','quiz','first');await jobStorage.claim(owner,'locked','quiz','second');await jobStorage.release(owner,'locked','quiz','second');
});
test('short-answer questions preserve equivalent accepted answers and operators instead of unsafe fuzzy math grading',()=>{
 const question=parseQuiz(JSON.stringify([{type:'short-answer',question:'SI unit of force?',answer:'Newton',acceptedAnswers:['newtons','N'],hint:'Use the SI name',explanation:'Force is measured in newtons.'}]))[0];expect(isQuizCorrect(question,' NEWTON. ')).toBe(true);expect(isQuizCorrect(question,'n')).toBe(true);expect(isQuizCorrect(question,'joule')).toBe(false);expect(isQuizCorrect({...question,answer:'x > 2',acceptedAnswers:[]},'x < 2')).toBe(false);
 expect(()=>parseQuiz(JSON.stringify([{...question,acceptedAnswers:[]}]))).toThrow('incomplete');
});

test('an applied draft and a stale job ID cannot be resumed and appended twice',async()=>{
 const options={fileId:'once',kind:'flashcards',input:{},total:1,step:jest.fn(async()=>({question:'A unique card'}))};await runCheckpointJob(options);const saved=await jobStorage.get(owner,'once','flashcards');const patch=appliedGenerationPatch({id:'once'},'flashcards',saved);while((await jobStorage.get(owner,'once','flashcards')).status!=='completed')await new Promise(resolve=>setTimeout(resolve,0));
 await expect(runCheckpointJob({...options,resume:saved})).rejects.toThrow('already saved');expect(patch.generationJobs.flashcards.status).toBe('completed');expect(options.step).toHaveBeenCalledTimes(1);
 await jobStorage.put({...saved,id:'newer-job',status:'paused',updatedAt:Date.now()+100});await expect(runCheckpointJob({...options,resume:saved})).rejects.toThrow('newer generation job');expect((await jobStorage.get(owner,'once','flashcards')).id).toBe('newer-job');
});
