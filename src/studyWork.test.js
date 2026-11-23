import {aiOwner,setAiIdentityProvider} from './aiAccess';
import {attachStudyWorkspace,startStudyWork,cancelStudyWork,isStudyWorkRunning} from './studyWork';
let detach;
beforeEach(()=>{setAiIdentityProvider(()=>({uid:'work-'+expect.getState().currentTestName}));});
afterEach(()=>{detach?.();cancelStudyWork();});
test('background completion merges into the live file after navigation and concurrent edits, and deduplicates work',async()=>{
 let file={id:'one',name:'Pack',notes:'Original',linkedFileIds:['key']},finish;const gate=new Promise(resolve=>{finish=resolve;});let calls=0;
 detach=attachStudyWorkspace({owner:aiOwner(),get:id=>id===file.id?file:null,update:(_id,patch)=>{file=patch(file);}});
 const options={file,kind:'notes',work:async context=>{calls++;await gate;context.commit(current=>({...current,notes:current.notes+'\nGenerated'}));return true;}};
 const first=startStudyWork(options),second=startStudyWork(options);expect(second).toBe(first);expect(isStudyWorkRunning(file.id)).toBe(true);file={...file,notes:'Edited while generating'};finish();await first.promise;expect(calls).toBe(1);expect(isStudyWorkRunning(file.id)).toBe(false);expect(file).toMatchObject({notes:'Edited while generating\nGenerated',linkedFileIds:['key']});
});
test('cancel, deletion and account changes prevent late results from modifying a workspace',async()=>{
 let file={id:'one',name:'Pack',notes:'Keep'},finish;detach=attachStudyWorkspace({owner:aiOwner(),get:()=>file,update:(_id,patch)=>{file=patch(file);}});
 const task=startStudyWork({file,kind:'notes',work:async context=>{await new Promise(resolve=>{finish=resolve;});context.commit(current=>({...current,notes:'Late'}));}});await Promise.resolve();cancelStudyWork(file.id);finish();await expect(task.promise).rejects.toHaveProperty('name','AbortError');expect(file.notes).toBe('Keep');
 let finishOther;const next=startStudyWork({file,kind:'notes',work:async context=>{await new Promise(resolve=>{finishOther=resolve;});context.commit(current=>({...current,notes:'Wrong account'}));}});await Promise.resolve();setAiIdentityProvider(()=>({uid:'another'}));finishOther();await expect(next.promise).rejects.toHaveProperty('name','AbortError');expect(file.notes).toBe('Keep');
});
