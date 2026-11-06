const {test}=require('node:test'),assert=require('node:assert/strict');
const admin=require('./admin'),quota=require('./ai-quota'),agent=require('../api/agent');
const {TERMS_VERSION}=require('./access');
const unit={title:'Energy',lesson:'Energy transfers between objects.',question:'Can energy transfer?',options:['Yes','Never','Only at night'],correct:0,explanation:'Objects transfer energy through interactions.'};
function body(tool='create_notes'){return {tool,requestId:'12345678-1234-1234-1234-123456789abc',confirmed:true,arguments:{fileId:'physics',title:'Energy notes',instructions:'Use my source.',count:2,difficulty:'beginner',language:'English'}};}
function setup(context,output){
 process.env.GROQ_API_KEY='test-only-groq-key';
 const values=new Map([['users/alice/access/verification',{stage:'accepted',adult:true,version:TERMS_VERSION}],['users/alice/files/physics',{id:'physics',name:'Physics',notes:'Energy transfers between objects.',folderId:'school'}],['users/alice/folders/school',{id:'school',name:'School'}],['users/bob/files/private',{notes:'Private Bob notes',folderId:'secret'}]]);
 context.mock.method(admin,'identity',()=>({verifyIdToken:async()=>({uid:'alice',email_verified:true,firebase:{sign_in_provider:'google.com'}})}));
 const reference=path=>({path,get:async()=>({data:()=>values.get(path)}),set:async(value,options)=>{values.set(path,options?.merge?{...values.get(path),...value}:value);}});
 let fail=false;
 context.mock.method(admin,'database',()=>({doc:reference,runTransaction:async operation=>{const writes=[];const value=await operation({get:ref=>ref.get(),set:(ref,data)=>writes.push([ref.path,data])});if(fail&&writes.some(([path])=>path.includes('/files/agent_')))throw new Error('private service configuration');for(const [path,data] of writes)values.set(path,data);return value;}}));
 context.mock.method(quota,'configured',()=>true);const reservations=context.mock.method(quota,'reserveRequest',async()=>({}));
 const requests=context.mock.method(global,'fetch',async(_url,options)=>({ok:true,json:async()=>({choices:[{message:{content:typeof output==='function'?output(JSON.parse(options.body),values):output}}]})}));
 return {values,requests,reservations,failWrites:()=>{fail=true;}};
}
async function invoke(payload=body(),signed=true){const result={};await agent({method:'POST',headers:{'content-type':'application/json',...(signed?{authorization:'Bearer local-test-token'}:{})},body:payload},{setHeader(){},status(value){result.status=value;return this;},json(value){result.body=value;}});return result;}

test('planning uses owned source, returns a reviewable action and never executes it',async context=>{
 const state=setup(context,JSON.stringify({tool:'create_notes',title:'Energy notes',instructions:'Explain the supplied source.',difficulty:'beginner',count:2,message:'I saved it!'}));
 const result=await invoke({tool:'plan_action',requestId:body().requestId,arguments:{fileId:'physics',prompt:'Make notes from this file.'}});
 assert.equal(result.status,200);assert.equal(result.body.saved,false);assert.equal(result.body.action.arguments.fileId,'physics');assert.doesNotMatch(result.body.message,/saved/i);assert.equal(state.values.size,4);
});
for(const [tool,output,field] of [['create_notes','# Energy\n\nEnergy transfers.','notes'],['create_flashcards',JSON.stringify([{question:'What transfers?',answer:'Energy.'},{question:'Between what?',answer:'Objects.'}]),'studyCards'],['create_quiz',JSON.stringify([unit,unit]),'studyQuiz'],['create_course',JSON.stringify([unit,unit]),'course']]){
 test(tool+' creates a separate confirmed file without overwriting the source',async context=>{
  const state=setup(context,output);const result=await invoke(body(tool));assert.equal(result.status,200);assert.equal(result.body.saved,true);assert(result.body.file[field]);assert.equal(result.body.file.folderId,'school');assert.equal(state.values.get('users/alice/files/physics').notes,'Energy transfers between objects.');assert.equal(state.values.get('users/bob/files/private').notes,'Private Bob notes');
 });
}
test('workspace tools reject anonymous access, another owner’s file, and missing confirmation before generation',async context=>{
 const state=setup(context,'# Notes');assert.equal((await invoke(body(),false)).status,401);const other=body();other.arguments.fileId='private';assert.equal((await invoke(other)).status,404);assert.equal((await invoke({...body(),confirmed:false})).status,400);assert.equal((await invoke({...body(),tool:'execute_shell'})).status,400);assert.equal(state.requests.mock.callCount(),0);
});
test('successful action retries reuse a completed result without another provider request or quota reservation',async context=>{
 const state=setup(context,'# Energy\n\nAn explanation.');assert.equal((await invoke()).status,200);assert.equal((await invoke()).status,200);assert.equal(state.requests.mock.callCount(),1);assert.equal(state.reservations.mock.callCount(),1);const changed=body();changed.arguments.title='Changed';assert.equal((await invoke(changed)).status,409);
});
test('an unsupported destructive proposal is rejected without writes',async context=>{
 const state=setup(context,JSON.stringify({tool:'delete_file',title:'Delete everything',instructions:'',difficulty:'beginner',count:2}));const result=await invoke({tool:'plan_action',requestId:body().requestId,arguments:{fileId:'physics',prompt:'Delete files'}});assert.equal(result.status,502);assert.equal(state.values.size,4);
});
test('invalid generated content and failed transactions never acknowledge saving',async context=>{
 const state=setup(context,'not a JSON quiz');let result=await invoke(body('create_quiz'));assert.equal(result.status,502);assert.notEqual(result.body.saved,true);
 state.failWrites();result=await invoke({...body(),requestId:'22345678-1234-1234-1234-123456789abc'});assert.equal(result.status,503);assert.notEqual(result.body.saved,true);assert.doesNotMatch(JSON.stringify(result.body),/private service/);assert.equal([...state.values.keys()].filter(key=>key.includes('/files/agent_')).length,0);
});
test('a source changed during generation is preserved and cannot produce a stale confirmed derivative',async context=>{
 const state=setup(context,(_input,values)=>{values.set('users/alice/files/physics',{...values.get('users/alice/files/physics'),notes:'A newer edit.'});return '# Energy notes';});const result=await invoke();assert.equal(result.status,409);assert.equal(state.values.get('users/alice/files/physics').notes,'A newer edit.');assert.notEqual(result.body.saved,true);
});
