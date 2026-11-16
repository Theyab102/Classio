const {test}=require('node:test');
const assert=require('node:assert/strict');
const {imageOperation}=require('./image-service');
test('cutout validation rejects spoofed image bytes before contacting the provider',async()=>{
 let calls=0;await assert.rejects(imageOperation({action:'remove-background',image:'data:image/png;base64,YmFk'},{env:{REMOVE_BG_API_KEY:'fixture'},fetcher:async()=>{calls++;throw Error('should not call');}}),error=>error.status===400);
 assert.equal(calls,0);
});
test('cutout credit/auth/rate errors have actionable messages without leaking provider bodies',async()=>{
 const image='data:image/png;base64,'+Buffer.from('89504e470d0a1a0a0000000000000000','hex').toString('base64');
 for(const [status,pattern] of [[401,/credential/i],[402,/credit/i],[429,/limit/i]])await assert.rejects(imageOperation({action:'remove-background',image},{env:{REMOVE_BG_API_KEY:'fixture'},fetcher:async()=>({ok:false,status,json:async()=>({secret:'never expose'})})}),error=>pattern.test(error.message)&&!error.message.includes('never expose'));
});
