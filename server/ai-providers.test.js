const {test,beforeEach}=require('node:test');
const assert=require('node:assert/strict');
const {generateStudy,availableProviders,clearProviderCaches,geminiPayload,providerHealth}=require('./ai-providers');
const {providerCredentials}=require('./ai-settings');
const {freeOpenRouter}=require('./ai-adapters');
const {providerPolicy}=require('./ai-policy');
beforeEach(clearProviderCaches);
const env={Groq_Key:'fixture-groq-primary',Groq_Key_packup:'fixture-groq-backup',AI_GROQ_STUDENT_USE_APPROVED:'true',AI_GROQ_FREE_TIER_CONFIRMED:'true',CLOUDFLARE_API_TOKEN:'fixture-cf-primary',CLOUDFLARE_ACCOUNT_ID:'a'.repeat(32),AI_CLOUDFLARE_STUDENT_USE_APPROVED:'true',AI_CLOUDFLARE_FREE_TIER_CONFIRMED:'true'};
const input={messages:[{role:'system',content:'Use the learner’s language.'},{role:'user',content:'Explain gravity.'}],maxTokens:500};
const response=(data,status=200,headers={})=>({ok:status>=200&&status<300,status,headers:{get:name=>headers[name]},json:async()=>data});
const catalog=response({data:[{id:'llama-3.1-8b-instant',context_window:131072},{id:'llama-3.3-70b-versatile',context_window:131072},{id:'whisper-large-v3-turbo'}]});
const cfCatalog=response({result:[{name:'@cf/meta/llama-3.1-8b-instruct-fp8',task:{name:'Text Generation'},properties:{context_window:8192}}]});
function upstream(run){return async(url,options)=>{if(url.endsWith('/models'))return catalog;if(url.endsWith('/models/search'))return cfCatalog;return run(url,options);};}
const answered=text=>response({choices:[{message:{content:text},finish_reason:'stop'}]});

test('exact packup names win over legacy aliases and empty optional backups are valid',()=>{
 for(const [provider,primary,backup]of[['groq','Groq_Key','Groq_Key_packup'],['gemini','Gemini_Key','Gemini_Key_packup'],['openrouter','OPENROUTER_API_KEY','OPENROUTER_API_KEY_packup'],['cloudflare','CLOUDFLARE_API_TOKEN','CLOUDFLARE_API_TOKEN_packup'],['ollama','OLLAMA_API_TOKEN','OLLAMA_API_TOKEN_packup']]){
  assert.deepEqual(providerCredentials(provider,{[primary]:'fixture-primary',[backup]:'fixture-backup'}).map(key=>[key.name,key.role]),[[primary,'primary'],[backup,'backup']]);assert.equal(providerCredentials(provider,{[primary]:'fixture-primary'}).length,1);
 }
 assert.equal(providerCredentials('gemini',{Gemini_Key:'fixture-exact',GEMINI_API_KEY:'fixture-alias',Gemini_Key_packup:'fixture-backup',GEMINI_API_KEY_2:'fixture-alias-backup'})[1].name,'Gemini_Key_packup');
 assert.equal(providerCredentials('groq',{Groq_Key:'fixture-same',Groq_Key_packup:'fixture-same'}).length,1);
});
test('Cloudflare backup tokens are paired with the correct account and default to the primary account',()=>{
 const settings={...env,CLOUDFLARE_API_TOKEN_packup:'fixture-cf-backup'};assert.equal(providerCredentials('cloudflare',settings)[1].accountId,settings.CLOUDFLARE_ACCOUNT_ID);
 settings.CLOUDFLARE_ACCOUNT_ID_packup='b'.repeat(32);assert.equal(providerCredentials('cloudflare',settings)[1].accountId,'b'.repeat(32));
});
test('Gemini is blocked for minors regardless of old enablement flags and all other providers require eligibility/free confirmations',()=>{
 assert.equal(providerPolicy('gemini',{Gemini_Key:'fixture',GEMINI_ENABLED:'true',AI_ALLOW_PAID:'true'}).enabled,false);
 assert.deepEqual(availableProviders({Groq_Key:'fixture'}),[]);assert.deepEqual(availableProviders(env),['groq','cloudflare']);
 assert.equal(providerPolicy('groq',{Groq_Key:'fixture',AI_GROQ_STUDENT_USE_APPROVED:'true'}).enabled,false);
});
test('successful primary text generation never touches a backup and diagnostics contain no values or prompts',async()=>{
 const calls=[],reports=[];const result=await generateStudy(input,{env,report:event=>reports.push(event),fetcher:upstream(async(url,options)=>{calls.push(options.headers.Authorization);assert.deepEqual(JSON.parse(options.body).messages,input.messages);return answered('Gravity attracts masses.');})});
 assert.equal(result.provider,'groq');assert.deepEqual(calls,['Bearer fixture-groq-primary']);assert(!JSON.stringify(reports).includes('fixture-'));assert(!JSON.stringify(reports).includes('gravity'));
});
for(const status of [401,403])test('only credential rejection '+status+' uses a backup; invalid primary stays in cooldown',async()=>{
 const calls=[],options={env,fetcher:upstream(async(url,request)=>{calls.push(request.headers.Authorization);return request.headers.Authorization.endsWith('primary')?response({},status):answered('Backup answer');})};
 assert.equal((await generateStudy(input,options)).text,'Backup answer');await generateStudy({...input,maxTokens:501},options);assert.deepEqual(calls,['Bearer fixture-groq-primary','Bearer fixture-groq-backup','Bearer fixture-groq-backup']);
});
test('a rejected Cloudflare primary uses the backup account ID as well as its token',async()=>{
 const settings={...env,Groq_Key:'',Groq_Key_packup:'',CLOUDFLARE_API_TOKEN_packup:'fixture-cf-backup',CLOUDFLARE_ACCOUNT_ID_packup:'b'.repeat(32)},seen=[];
 const result=await generateStudy(input,{env:settings,fetcher:async(url,options)=>{seen.push([url,options.headers.Authorization]);if(url.includes('/accounts/'+ 'a'.repeat(32)))return response({},403);return url.endsWith('/search')?cfCatalog:response({result:{response:'Paired backup'}});}});
 assert.equal(result.text,'Paired backup');assert(seen.at(-1)[0].includes('b'.repeat(32)));assert.equal(seen.at(-1)[1],'Bearer fixture-cf-backup');
});
test('shared 429 respects Retry-After, skips same-provider backup, and falls back to a different eligible provider',async()=>{
 const seen=[];const options={env,fetcher:upstream(async(url,request)=>{seen.push(request.headers.Authorization);return url.includes('groq')?response({},429,{'retry-after':'90'}):response({result:{response:'Free fallback'}});})};
 assert.equal((await generateStudy(input,options)).provider,'cloudflare');await generateStudy({...input,maxTokens:501},options);assert.deepEqual(seen,['Bearer fixture-groq-primary','Bearer fixture-cf-primary','Bearer fixture-cf-primary']);assert.equal(providerHealth().groq.status,'temporarily-limited');
});
for(const failure of ['network',500,503])test('temporary failure '+failure+' falls back without credential cycling',async()=>{
 const seen=[];const result=await generateStudy(input,{env,fetcher:upstream(async(url,request)=>{seen.push(request.headers.Authorization);if(url.includes('groq')){if(failure==='network')throw Error('sensitive upstream body');return response({},failure);}return response({result:{response:'Recovered'}});})});assert.equal(result.text,'Recovered');assert(!seen.includes('Bearer fixture-groq-backup'));
});
test('permanent invalid requests and content policy blocks do not bypass the error through another provider',async()=>{
 for(const status of [400,422]){clearProviderCaches();let calls=0;await assert.rejects(generateStudy(input,{env,fetcher:upstream(async()=>{calls++;return response({},status);})}));assert.equal(calls,1);}
 let calls=0;await assert.rejects(generateStudy(input,{env,fetcher:upstream(async()=>{calls++;return response({choices:[{finish_reason:'content_filter'}]});})}),error=>error.code==='CONTENT_BLOCKED');assert.equal(calls,1);
});
test('truncated output gets one alternate listed model; partial text is never returned',async()=>{
 const models=[];const result=await generateStudy(input,{env,fetcher:upstream(async(url,request)=>{models.push(JSON.parse(request.body).model);return models.length===1?response({choices:[{message:{content:'truncated'},finish_reason:'length'}]}):answered('Complete answer');})});assert.equal(result.text,'Complete answer');assert.deepEqual(models,['llama-3.1-8b-instant','llama-3.3-70b-versatile']);
});
test('an exhausted model list retains OUTPUT_LIMIT for the resumable client',async()=>{
 await assert.rejects(generateStudy(input,{env:{...env,CLOUDFLARE_API_TOKEN:'',AI_GROQ_MODEL:'llama-3.1-8b-instant'},fetcher:upstream(async()=>response({choices:[{finish_reason:'length'}]}))}),error=>error.code==='OUTPUT_LIMIT');
});
for(const stalled of ['catalog','body','request'])test('bounded primary timeout includes '+stalled+' and leaves time for fallback',async()=>{
 let primarySignal;
 const result=await generateStudy(input,{env:{...env,AI_PRIMARY_TIMEOUT_MS:'1000'},fetcher:async(url,options)=>{
  if(url.includes('groq')){primarySignal=options.signal;if(stalled==='catalog'||stalled==='request'&&!url.endsWith('/models'))return new Promise(()=>{});if(url.endsWith('/models'))return catalog;return{ok:true,json:()=>new Promise(()=>{})};}
  return url.endsWith('/search')?cfCatalog:response({result:{response:'Deadline fallback'}});
 }});assert.equal(result.text,'Deadline fallback');assert(primarySignal.aborted);
});
test('circuit cooldown expires without being extended by skipped requests',async()=>{
 const now=Date.now,base=now();let offset=0,primary=0;Date.now=()=>base+offset;
 try{const options={env,fetcher:upstream(async url=>{if(url.includes('groq')){if(++primary===1)throw Error('offline');return answered('Restored');}return response({result:{response:'Fallback'}});})};assert.equal((await generateStudy(input,options)).provider,'cloudflare');offset=5000;assert.equal((await generateStudy({...input,maxTokens:501},options)).provider,'cloudflare');offset=11000;assert.equal((await generateStudy({...input,maxTokens:502},options)).provider,'groq');assert.equal(primary,2);}finally{Date.now=now;}
});
test('style request deduplication/cache is private to the authorized account',async()=>{
 let calls=0;const options={env,identity:'user:a',fetcher:upstream(async url=>{calls++;return url.includes('cloudflare')?response({result:{response:'style'}}):answered('style');})};
 await Promise.all([generateStudy({...input,task:'style'},options),generateStudy({...input,task:'style'},options)]);await generateStudy({...input,task:'style'},options);assert.equal(calls,1);await generateStudy({...input,task:'style'},{...options,identity:'user:b'});assert.equal(calls,2);
});
test('audio only chooses a listed transcription model and never falls back to text-only providers',async()=>{
 const result=await generateStudy({audio:'dGVzdA==',filename:'lecture.webm'},{env,fetcher:upstream(async(url,request)=>{assert(url.endsWith('/audio/transcriptions'));assert.equal(request.body.get('model'),'whisper-large-v3-turbo');return response({text:'Transcript'});})});assert.equal(result.text,'Transcript');
 await assert.rejects(generateStudy({audio:'dGVzdA==',filename:'lecture.webm'},{env:{...env,Groq_Key:'',Groq_Key_packup:''},fetcher:()=>assert.fail('text-only provider must not receive audio')}),error=>error.code==='NOT_CONFIGURED');
});
test('vision chooses only explicitly capable models and supports OpenRouter vision payloads',async()=>{
 const settings={OPENROUTER_API_KEY:'fixture-router',AI_OPENROUTER_STUDENT_USE_APPROVED:'true'};
 const vision={...input,vision:true,messages:[{role:'user',content:[{type:'text',text:'Explain this diagram.'},{type:'image_url',image_url:{url:'data:image/png;base64,dGVzdA=='}}]}]};
 const result=await generateStudy(vision,{env:settings,fetcher:async(url,request)=>url.endsWith('/models')?response({data:[{id:'fixture/vision:free',context_length:8192,architecture:{input_modalities:['text','image']},pricing:{prompt:'0',completion:'0',image:'0'}}]}):(assert.deepEqual(JSON.parse(request.body).messages,vision.messages),answered('Diagram explanation'))});assert.equal(result.provider,'openrouter');
});
test('free-only OpenRouter verifies every catalog price and constrains routing; a free-looking paid ID cannot run',async()=>{
 assert(!freeOpenRouter({id:'bad:free',pricing:{prompt:'0',completion:'0',request:'0.01'}}));assert(!freeOpenRouter({id:'paid',pricing:{prompt:'0',completion:'0'}}));
 const settings={OPENROUTER_API_KEY:'fixture-router',AI_OPENROUTER_STUDENT_USE_APPROVED:'true'},models=[{id:'paid:free',context_length:8192,pricing:{prompt:'1',completion:'0'}},{id:'openrouter/free',context_length:8192,pricing:{prompt:'0',completion:'0',request:'0'}}];let body;
 await generateStudy(input,{env:settings,fetcher:async(url,request)=>url.endsWith('/models')?response({data:models}):(body=JSON.parse(request.body),answered('Free'))});assert.equal(body.model,'openrouter/free');assert.deepEqual(body.provider.max_price,{prompt:0,completion:0});assert.equal(body.provider.allow_fallbacks,false);
 await assert.rejects(generateStudy(input,{env:{...settings,AI_OPENROUTER_MODEL:'paid:free'},fetcher:async()=>assert.fail('cached catalog must reject paid model')}),error=>error.code==='MODEL_UNAVAILABLE');
});
test('Ollama is optional, requires a protected remote endpoint/license approval and uses its actual capability catalog',async()=>{
 assert(!providerPolicy('ollama',{AI_ENABLE_OLLAMA:'true',AI_OLLAMA_STUDENT_USE_APPROVED:'true',OLLAMA_BASE_URL:'http://localhost:11434',VERCEL:'1'}).enabled);
 const settings={AI_ENABLE_OLLAMA:'true',AI_OLLAMA_STUDENT_USE_APPROVED:'true',AI_OLLAMA_LICENSE_APPROVED:'true',AI_OLLAMA_FREE_TIER_CONFIRMED:'true',OLLAMA_BASE_URL:'https://private.example.com',OLLAMA_MODEL:'qwen2.5:1.5b',OLLAMA_API_TOKEN:'fixture-private-token'};
 const seen=[];const result=await generateStudy(input,{env:settings,fetcher:async(url,request)=>{seen.push(url);assert.equal(request.headers.Authorization,'Bearer fixture-private-token');return response(url.endsWith('/tags')?{models:[{name:settings.OLLAMA_MODEL}]}:url.endsWith('/show')?{capabilities:['completion'],model_info:{'qwen2.context_length':32768}}:{message:{content:'Private answer'},done_reason:'stop'});}});assert.equal(result.text,'Private answer');assert.equal(seen.length,3);
});
test('Gemini payload conversion retains mathematics, separate system instructions and inline images while activation is blocked',()=>{
 const payload=geminiPayload({...input,messages:[{role:'system',content:'Use $x^2$.'},{role:'user',content:[{type:'text',text:'Diagram?'},{type:'image_url',image_url:{url:'data:image/png;base64,dGVzdA=='}}]}]});assert.equal(payload.systemInstruction.parts[0].text,'Use $x^2$.');assert.equal(payload.contents[0].parts[1].inlineData.data,'dGVzdA==');
});

test('normal image bytes use a visual token budget rather than being counted as base64 text tokens',async()=>{
 const vision={...input,vision:true,messages:[{role:'user',content:[{type:'image_url',image_url:{url:'data:image/jpeg;base64,'+Buffer.alloc(600000,23).toString('base64')}},{type:'text',text:'Explain this diagram.'}]}]};
 const result=await generateStudy(vision,{env,fetcher:async(url,request)=>url.endsWith('/models')?response({data:[{id:'meta-llama/llama-4-scout-17b-16e-instruct',context_window:131072}]}):(assert.equal(JSON.parse(request.body).messages[0].content[0].image_url.url,vision.messages[0].content[0].image_url.url),answered('The diagram shows motion.'))});assert.equal(result.provider,'groq');
});

test('client Auto respects the configured server routing preference',async()=>{
 const result=await generateStudy({...input,provider:'auto'},{env:{...env,AI_ROUTING_MODE:'cloudflare'},fetcher:upstream(async url=>{
  assert(url.includes('cloudflare'));return response({result:{response:'Server preference'}});
 })});
 assert.equal(result.provider,'cloudflare');
});

test('free-only routing rejects unknown or empty catalog prices instead of coercing them to zero',()=>{
 for(const pricing of [{prompt:null,completion:'0'},{prompt:'0',completion:''},{prompt:'0',completion:'0',image:null}])assert.equal(freeOpenRouter({id:'fixture:free',pricing}),false);
});
