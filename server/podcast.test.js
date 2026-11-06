const {test,beforeEach,afterEach}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),crypto=require('node:crypto');
const {setAccess}=require('./access');
const {podcastFingerprint,podcastInput}=require('../src/shared/podcast-schema');
let quota,ready,reservations,previous;
const moduleObject={exports:{}};
vm.runInNewContext(fs.readFileSync(require.resolve('../api/podcast'),'utf8'),{module:moduleObject,exports:moduleObject.exports,process,Buffer,Date,require:name=>name==='../server/ai-quota'?{configured:()=>ready,reserveRequest:async()=>{reservations++;return quota;}}:require(name.startsWith('../server/')?'./'+name.slice('../server/'.length):name)});
const podcast=moduleObject.exports;
beforeEach(()=>{previous={secret:process.env.CLASSIO_TTS_SIGNING_SECRET,session:process.env.CLASSIO_SESSION_SECRET};process.env.CLASSIO_TTS_SIGNING_SECRET='test-only-podcast-signing-secret-32-characters';process.env.CLASSIO_SESSION_SECRET='test-only-session-signing-secret-over-32-chars';quota={};ready=true;reservations=0;});
afterEach(()=>{for(const [key,value] of [['CLASSIO_TTS_SIGNING_SECRET',previous.secret],['CLASSIO_SESSION_SECRET',previous.session]]){if(value===undefined)delete process.env[key];else process.env[key]=value;}});
const payload=()=>({language:'en',voices:['aria','echo'],speed:1,dialogue:[{speaker:0,text:'Energy is conserved.'},{speaker:1,text:'How does energy transfer?'}]});
async function invoke(changes={}){const result={headers:{}};let cookie;setAccess({setHeader:(_name,value)=>{cookie=value.split(';')[0];}},{stage:'accepted',adult:true});const req={method:'POST',headers:{'content-type':'application/json',cookie},body:payload(),...changes};await podcast(req,{setHeader:(name,value)=>{result.headers[name]=value;},status:code=>{result.status=code;return{json:body=>{result.body=body;}};}});return result;}

test('podcast ticket is short-lived, signed and bound to normalized script without exposing the signing key',async()=>{
 const result=await invoke();assert.equal(result.status,200);assert.equal(reservations,1);
 const [body,signature]=result.body.ticket.split('.');const claim=JSON.parse(Buffer.from(body,'base64url'));
 assert.equal(claim.h,crypto.createHash('sha256').update(podcastFingerprint(podcastInput(payload()))).digest('hex'));
 assert.match(claim.o,/^[a-f0-9]{64}$/);assert(claim.exp*1000>Date.now()&&claim.exp*1000<=Date.now()+601000);
 assert.equal(signature,crypto.createHmac('sha256',process.env.CLASSIO_TTS_SIGNING_SECRET).update(body).digest('base64url'));
 assert(!JSON.stringify(result.body).includes(process.env.CLASSIO_TTS_SIGNING_SECRET));
});
test('missing eligibility and cross-site callers never consume audio quota',async()=>{
 assert.equal((await invoke({headers:{'content-type':'application/json'}})).status,403);
 assert.equal((await invoke({headers:{'content-type':'application/json','sec-fetch-site':'cross-site'}})).status,403);
 assert.equal(reservations,0);
});
test('unconfigured signing and quota exhaustion fail with recoverable configuration/limit errors',async()=>{
 process.env.CLASSIO_TTS_SIGNING_SECRET='';assert.equal((await invoke()).status,503);assert.equal(reservations,0);
 process.env.CLASSIO_TTS_SIGNING_SECRET='test-only-podcast-signing-secret-32-characters';quota={retryAfter:60};const result=await invoke();assert.equal(result.status,429);assert.equal(result.headers['Retry-After'],'60');assert.equal(result.body.ticket,undefined);
});
test('invalid dialogue, unsupported language and client identity fields are rejected before authorization',async()=>{
 for(const patch of [{voices:['aria','aria']},{language:'ar'},{owner:'someone-else'},{dialogue:[{speaker:2,text:'Wrong speaker'}]}])assert.equal((await invoke({body:{...payload(),...patch}})).status,400);
 assert.equal(reservations,0);
});
