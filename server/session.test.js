const {test,after}=require('node:test'),assert=require('node:assert/strict');
let quotaCalls=0,quota={};
require.cache[require.resolve('./ai-quota')]={exports:{reserveRequest:async()=>{quotaCalls++;return quota;}}};
const session=require('../api/session');
const old=process.env.CLASSIO_SESSION_SECRET;process.env.CLASSIO_SESSION_SECRET='fixture-session-secret-at-least-32-characters';after(()=>{if(old===undefined)delete process.env.CLASSIO_SESSION_SECRET;else process.env.CLASSIO_SESSION_SECRET=old;});
async function invoke(headers={},method='POST'){const result={headers:{}};await session({method,headers:{host:'classio.example','content-type':'application/json',...headers}},{setHeader:(name,value)=>result.headers[name]=value,status(code){result.status=code;return this;},json(value){result.body=value;}});return result;}
test('guest page refresh reuses the signed identity without exhausting new-session quotas',async()=>{
 quotaCalls=0;quota={};const first=await invoke();assert.equal(first.status,200);assert.equal(quotaCalls,1);quota={retryAfter:3600};const again=await invoke({cookie:first.headers['Set-Cookie'].split(';')[0]});assert.equal(again.status,200);assert.equal(again.body.csrf,first.body.csrf);assert.equal(quotaCalls,1);assert.equal((await invoke()).status,429);
});
test('guest issuance rejects cross-origin and invalid methods/content before quota or signing',async()=>{
 quotaCalls=0;assert.equal((await invoke({},'GET')).status,405);assert.equal((await invoke({'content-type':'text/plain'})).status,415);assert.equal((await invoke({origin:'https://evil.example'})).status,403);assert.equal(quotaCalls,0);
});
