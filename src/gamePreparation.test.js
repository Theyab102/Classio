import 'fake-indexeddb/auto';
import {aiOwner,setAiIdentityProvider} from './aiAccess';
import {attachStudyWorkspace,cancelStudyWork} from './studyWork';
import {prepareGameCards} from './gamePreparation';
import {sendStudyRequest} from './studyTools';
import {jobStorage} from './generationJobs';
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
jest.mock('./studyTools',()=>({...jest.requireActual('./studyTools'),sendStudyRequest:jest.fn()}));
let file,detach;
beforeEach(()=>{setAiIdentityProvider(()=>({uid:'game-'+expect.getState().currentTestName}));file={id:'one',name:'Source',notes:'Keep source text',linkedFileIds:['key']};detach=attachStudyWorkspace({owner:aiOwner(),get:()=>file,update:(_id,patch)=>{file=patch(file);}});jest.clearAllMocks();});
afterEach(()=>{detach();cancelStudyWork();});
const cards=()=>JSON.stringify(Array.from({length:10},(_,i)=>({question:'Question '+i,answer:'Answer '+i})));
test('game cards share the background job, preserve concurrent edits, and do not spend quota on duplicate preparation',async()=>{
 let finish;sendStudyRequest.mockImplementation(()=>new Promise(resolve=>{finish=()=>resolve(cards());}));const options={file,getFileText:async()=> 'Original source with equal groups'};const first=prepareGameCards(options),second=prepareGameCards(options);expect(second).toBe(first);
 while(!finish)await new Promise(resolve=>setTimeout(resolve,0));file={...file,notes:'Edited while games prepare',studyCards:[{id:'manual',question:'My card',answer:'Keep it'}]};finish();await first.promise;expect(file.studyCards).toHaveLength(11);expect(file).toMatchObject({notes:'Edited while games prepare',linkedFileIds:['key']});expect(file.studyCards[0].id).toBe('manual');expect(file.generationJobs.flashcards.status).toBe('completed');expect(sendStudyRequest).toHaveBeenCalledTimes(1);
});
test('opening games respects explicit cancellation and can resume on demand',async()=>{
 await jobStorage.put({id:'paused',version:1,owner:aiOwner(),file:file.id,kind:'flashcards',status:'paused',updatedAt:Date.now(),input:{source:'Original source',topic:'Source',count:10},sections:[]});
 await expect(prepareGameCards({file,automatic:true}).promise).rejects.toHaveProperty('name','AbortError');expect(sendStudyRequest).not.toHaveBeenCalled();sendStudyRequest.mockResolvedValue(cards());await prepareGameCards({file,automatic:false}).promise;expect(file.studyCards).toHaveLength(10);
});
test('late game preparation cannot replace deleted material',async()=>{
 let finish;sendStudyRequest.mockImplementation(()=>new Promise(resolve=>{finish=()=>resolve(cards());}));const task=prepareGameCards({file,getFileText:async()=> 'Source'});while(!finish)await new Promise(resolve=>setTimeout(resolve,0));file=null;cancelStudyWork('one');finish();await expect(task.promise).rejects.toHaveProperty('name','AbortError');expect(file).toBeNull();
});
