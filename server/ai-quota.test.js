const {test,beforeEach,after}=require('node:test'),assert=require('node:assert/strict');
const records=new Map();let queue=Promise.resolve();
const db={collection:()=>({doc:key=>({key})}),runTransaction:work=>{const run=queue.then(()=>work({getAll:async(...refs)=>refs.map(ref=>({data:()=>records.get(ref.key)})),set:(ref,value)=>records.set(ref.key,value)}));queue=run.catch(()=>{});return run;}};
require.cache[require.resolve('firebase-admin/app')]={exports:{cert:value=>value,getApps:()=>[{name:'classio-api'}],initializeApp:()=>({name:'classio-api'})}};
require.cache[require.resolve('firebase-admin/firestore')]={exports:{getFirestore:()=>db,Timestamp:{fromMillis:value=>value}}};
const {reserveRequest}=require('./ai-quota');
const fixture={VERCEL:'1',VERCEL_ENV:'production',AI_QUOTA_MODE:'firestore',CLASSIO_SESSION_SECRET:'fixture-shared-quota-secret-at-least-32-chars',FIREBASE_SERVICE_ACCOUNT_JSON:JSON.stringify({project_id:'fixture',client_email:'fixture@example.com',private_key:'fixture-private-key'})};
const old=Object.fromEntries(Object.keys(fixture).map(key=>[key,process.env[key]]));Object.assign(process.env,fixture);
after(()=>{for(const[key,value]of Object.entries(old))if(value===undefined)delete process.env[key];else process.env[key]=value;});
beforeEach(()=>records.clear());
const request=(user,ip='203.0.113.1')=>({headers:{'x-vercel-forwarded-for':ip},aiIdentity:{sub:'user:'+user}});
test('shared transactions enforce per-user and global concurrency across request instances, and release permits continuation',async()=>{
 const a=await reserveRequest(request('alice'),'study'),b=await reserveRequest(request('alice'),'study');assert.equal(typeof a.release,'function');assert.equal((await reserveRequest(request('alice'),'study')).retryAfter,15);
 const other=await Promise.all(Array.from({length:6},(_,i)=>reserveRequest(request('other-'+i),'study')));assert(other.every(result=>typeof result.release==='function'));assert.equal((await reserveRequest(request('ninth'),'study')).retryAfter,15);
 await a.release();assert.equal(typeof(await reserveRequest(request('ninth'),'study')).release,'function');await b.release();
});
test('persistent quotas survive module reloads, hash identities and block new guest IDs on one exhausted IP',async()=>{
 for(let i=0;i<40;i++)await reserveRequest(request('guest-'+i),'image-search');
 delete require.cache[require.resolve('./ai-quota')];const restarted=require('./ai-quota');assert((await restarted.reserveRequest(request('new-guest'),'image-search')).retryAfter>0);assert([...records.keys()].every(key=>!key.includes('guest-')&&!key.includes('203.0.113.1')));
});
test('expired concurrent leases are removed and missing trusted IP or quota configuration fails closed',async()=>{
 const now=Date.now,base=now();let offset=0;Date.now=()=>base+offset;try{await reserveRequest(request('alice'),'study');await reserveRequest(request('alice'),'study');offset=61000;assert.equal(typeof(await reserveRequest(request('alice'),'study')).release,'function');}finally{Date.now=now;}
 assert.deepEqual(await reserveRequest({headers:{},aiIdentity:{sub:'user:alice'}},'study'),{unavailable:true});
 const previous=process.env.FIREBASE_SERVICE_ACCOUNT_JSON;delete process.env.FIREBASE_SERVICE_ACCOUNT_JSON;try{assert.deepEqual(await reserveRequest(request('alice'),'study'),{unavailable:true});}finally{process.env.FIREBASE_SERVICE_ACCOUNT_JSON=previous;}
});
