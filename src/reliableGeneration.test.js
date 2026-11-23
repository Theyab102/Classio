import 'fake-indexeddb/auto';
import {setAiIdentityProvider} from './aiAccess';
import {generateNotes} from './checkpointGenerators';
import {runCheckpointJob,jobStorage,deleteGenerationJobs} from './generationJobs';
import {makeSourceBundle} from './sourceBundle';
import {retryDelay,acquireStudySlot} from './studyQueue';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
let owner;
beforeEach(()=>{owner='reliable-'+expect.getState().currentTestName;setAiIdentityProvider(()=>({uid:owner}));});
const request={messages:[{role:'system',content:'Teach at the source level.'},{role:'user',content:'Teach this pack.'}],maxTokens:6000};
const teaching=n=>`## Page ${n}\n\n### Question ${n}\nThe lesson is adding equal groups. First identify the number of groups, then multiply it by the size of each group. For this question, ${n} groups of two make ${n*2}. A worked example: three groups of two make six. Check by repeated addition.`;
test('all 33 PDF pages are taught, with every original question and linked answer key; no planning model call is spent',async()=>{
 const source=makeSourceBundle([{name:'Grade 11 revision.pdf',text:Array.from({length:33},(_,i)=>`[Page ${i+1}]\nQuestion ${i+1}: Find the size of ${i+1} groups of two.\n`).join('\n')},{name:'Revision answer key.txt',text:'Use repeated addition, then check the result. Question 33: 66.'}]);
 const send=jest.fn(async payload=>{const text=payload.messages[1].content,n=Number(/Coverage: section (\d+)/.exec(text)[1]);expect(text).toContain('Revision answer key.txt');expect(payload.messages[0].content).toContain('not exam logistics');expect(payload.messages[0].content).toContain('Do not assume that a grade number means calculus');return teaching(n);});
 const saved=[];const result=await generateNotes({fileId:'all-pages',input:{source,topic:'Grade 11 revision.pdf',coverageVersion:2,request},send,onCheckpoint:job=>saved.push(job)});
 expect(send).toHaveBeenCalledTimes(33);for(let n=1;n<=33;n++)expect(result).toContain(`### Question ${n}\n`);
 expect(saved.at(-1)).toMatchObject({status:'ready',total:33});expect(saved.at(-1).sections).toHaveLength(33);
});
test('diagram pages attach the original page to a vision-capable request while ordinary pages stay text-only',async()=>{
 const readPageImage=jest.fn(async()=> 'data:image/png;base64,dGVzdA=='),send=jest.fn(async payload=>teaching(Array.isArray(payload.messages[1].content)?1:2));
 const source=makeSourceBundle([{name:'Questions.pdf',text:'[Page 1]\nQuestion 1: Explain the graph shown below.\n\n[Page 2]\nQuestion 2: Add equal groups of two and explain how you check the sum.'}]);
 await generateNotes({fileId:'diagram',input:{source,topic:'Questions.pdf',coverageVersion:2,request},send,readPageImage});
 expect(readPageImage).toHaveBeenCalledTimes(1);expect(readPageImage).toHaveBeenCalledWith('Questions.pdf',1);expect(send.mock.calls[0][0].messages[1].content[0].type).toBe('image_url');expect(typeof send.mock.calls[1][0].messages[1].content).toBe('string');
});
test('a capacity reset retries only the unfinished section and renews the lease while waiting',async()=>{
 jest.useFakeTimers();const controller=new AbortController(),events=[],seen=[],saved=[];let limited=true;
 const store={claim:jest.fn(async()=>{}),release:jest.fn(async()=>{}),put:jest.fn(async job=>saved.push(job))};
 const pending=runCheckpointJob({fileId:'waiting',kind:'notes',input:{},total:3,signal:controller.signal,store,onCheckpoint:job=>events.push(job),step:async index=>{seen.push(index);if(index===1&&limited){limited=false;throw Object.assign(new Error('Capacity'),{status:429,retryAfter:45});}return teaching(index+1);}});
 const flush=async()=>{for(let i=0;i<50;i++)await Promise.resolve();};
 try{await flush();expect(seen).toEqual([0,1]);expect(events.at(-1)).toMatchObject({status:'waiting',total:3});expect(events.at(-1).sections).toHaveLength(1);const claims=store.claim.mock.calls.length;jest.advanceTimersByTime(20000);await flush();expect(store.claim.mock.calls.length).toBeGreaterThan(claims);jest.advanceTimersByTime(24999);await flush();expect(seen).toEqual([0,1]);jest.advanceTimersByTime(1);await flush();await expect(pending).resolves.toHaveLength(3);expect(seen).toEqual([0,1,1,2]);expect(saved.at(-1).status).toBe('ready');}finally{controller.abort();jest.useRealTimers();}
});
test('deleted material cannot be resurrected by a late provider response or paused checkpoint',async()=>{
 let release;const pending=runCheckpointJob({fileId:'deleted',kind:'notes',input:{},total:1,step:()=>new Promise(resolve=>{release=resolve;})});
 while(!release)await new Promise(resolve=>setTimeout(resolve,0));await deleteGenerationJobs('deleted');release(teaching(1));await expect(pending).rejects.toHaveProperty('name','AbortError');expect(await jobStorage.get(owner,'deleted','notes')).toBeUndefined();
});
test('daily and date-based resets are respected, and cancellation removes a waiting slot without blocking the next request',async()=>{
 expect(retryDelay('86400')).toBe(86400);expect(retryDelay(new Date(Date.now()+3600000).toUTCString())).toBeGreaterThan(3598);
 const first=await acquireStudySlot(owner),controller=new AbortController(),cancelled=acquireStudySlot(owner,controller.signal);const rejected=expect(cancelled).rejects.toHaveProperty('name','AbortError');controller.abort();await rejected;const next=acquireStudySlot(owner);first();const release=await next;release();
});
