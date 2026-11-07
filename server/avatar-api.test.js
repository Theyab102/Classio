const test=require('node:test'),assert=require('node:assert/strict'),Module=require('node:module'),fs=require('node:fs'),path=require('node:path');
const design={kind:'hat',region:'hat',baseStyle:2,name:'My blue cap',texture:'data:image/png;base64,'+fs.readFileSync(path.join(__dirname,'../public/favicon-192.png')).toString('base64')};
async function fixture(run){
 const before={enabled:process.env.AVATAR_SHARING_ENABLED,account:process.env.FIREBASE_SERVICE_ACCOUNT_JSON};process.env.AVATAR_SHARING_ENABLED='true';process.env.FIREBASE_SERVICE_ACCOUNT_JSON='fixture';
 const rows=new Map(),reports=[];let verified='google.com',quotaCalls=0;
 const collection={where(field,op,value){assert.equal(field,'status');assert.equal(op,'==');assert.equal(value,'approved');return{limit:n=>{assert.equal(n,24);return{get:async()=>({docs:[...rows].filter(([,data])=>data.status==='approved').map(([id,data])=>({id,data:()=>data}))})};}};},add:async data=>{const id='design-'+rows.size;rows.set(id,data);return{id};},doc:id=>({get:async()=>({id,exists:rows.has(id),data:()=>rows.get(id)})})};
 const load=Module._load,filename=require.resolve('../api/avatar-designs');delete require.cache[filename];
 Module._load=function(name,...args){if(name==='firebase-admin/app')return{getApps:()=>[{name:'classio-api'}]};if(name==='firebase-admin/auth')return{getAuth:()=>({verifyIdToken:async(token,revoked)=>{assert.equal(revoked,true);if(token==='invalid')throw Error('Invalid');return{uid:'creator',firebase:{sign_in_provider:verified}};}})};if(name==='firebase-admin/firestore')return{getFirestore:()=>({collection:name=>name==='avatarDesigns'?collection:{add:async data=>reports.push(data)}}),FieldValue:{serverTimestamp:()=>123}};if(name==='../server/ai-quota')return{reserveRequest:async()=>{quotaCalls++;return{};}};return load.call(this,name,...args);};
 let handler;try{handler=require(filename);}finally{Module._load=load;}
 async function request(body,headers={},method='POST'){const response={code:0,body:null,setHeader(){},status(code){this.code=code;return this;},json(body){this.body=body;return this;}};await handler({method,headers:{'content-type':'application/json','sec-fetch-site':'same-origin',...headers},body},response);return response;}
 try{await run({request,rows,reports,setProvider:value=>verified=value,quotaCalls:()=>quotaCalls});}finally{delete require.cache[filename];for(const[key,value]of[['AVATAR_SHARING_ENABLED',before.enabled],['FIREBASE_SERVICE_ACCOUNT_JSON',before.account]])if(value===undefined)delete process.env[key];else process.env[key]=value;}
}
test('sharing requires same-origin JSON and verified Google authentication',async()=>fixture(async({request,setProvider,quotaCalls})=>{
 assert.equal((await request({action:'submit',design},{'sec-fetch-site':'cross-site'})).code,403);assert.equal((await request({action:'submit',design})).code,401);assert.equal((await request({action:'submit',design},{authorization:'Bearer invalid'})).code,401);setProvider('password');assert.equal((await request({action:'submit',design},{authorization:'Bearer valid'})).code,403);assert.equal(quotaCalls(),0);
}));
test('submissions cannot bypass moderation and approved responses omit owner identifiers',async()=>fixture(async({request,rows})=>{
 const response=await request({action:'submit',design:{...design,visibility:'public',status:'approved'}},{authorization:'Bearer valid'});assert.equal(response.code,202);const item=rows.get(response.body.id);assert.equal(item.status,'pending');assert.equal(item.owner,'creator');assert.equal((await request(null,{},'GET')).body.designs.length,0);
 item.status='approved';const publicData=(await request(null,{},'GET')).body.designs[0];assert.equal(publicData.name,design.name);assert.equal(publicData.owner,undefined);assert.equal(publicData.status,undefined);
}));
test('reports are private records and reject designs that are not approved',async()=>fixture(async({request,rows,reports})=>{
 rows.set('private',{status:'pending',owner:'someone',design});assert.equal((await request({action:'report',id:'private',reason:'Please review this item'},{authorization:'Bearer valid'})).code,404);
 rows.get('private').status='approved';assert.equal((await request({action:'report',id:'private',reason:'Please review this item'},{authorization:'Bearer valid'})).code,202);assert.equal(reports[0].status,'open');assert.equal(reports[0].reporter,'creator');assert.equal(reports[0].designId,'private');
}));
