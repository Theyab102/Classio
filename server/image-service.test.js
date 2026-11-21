const test=require('node:test'),assert=require('node:assert/strict');
const {imageOperation,imageCapabilities}=require('./image-service');
const {sharedDesign}=require('./avatar-sharing');
const {searchImages,searchProviders,trackImageDownload}=require('./image-search');
const png='data:image/png;base64,'+Buffer.from('89504e470d0a1a0a0000000000000000','hex').toString('base64');
test('Gemini image inference stays blocked for student workspaces without contacting a provider',async()=>{
 const env={GEMINI_ENABLED:'true',Gemini_Key:'fixture-google',Gemini_Key_packup:'fixture-google-backup',GEMINI_IMAGE_MODEL:'fixture-model'};
 assert.equal(imageCapabilities(env).imageEditing,false);
 await assert.rejects(imageOperation({action:'avatar',prompt:'blue stripes',image:png},{env,fetcher:()=>assert.fail('must not activate Gemini')}),/student workspace/);
});
test('real cutout bytes are required; errors preserve original design',async()=>{
 const env={REMOVE_BG_API_KEY:'fixture-cutout-key',REMOVE_BG_FREE_TIER_CONFIRMED:'true'};
 const result=await imageOperation({action:'remove-background',image:png},{env,fetcher:async(url,options)=>{assert.match(url,/api.remove.bg/);assert.equal(options.body.get('image_file_b64'),png.split(',')[1]);return{ok:true,arrayBuffer:async()=>Buffer.from(png.split(',')[1],'base64')};}});assert.equal(result.image,png);
 await assert.rejects(imageOperation({action:'remove-background',image:png},{env,fetcher:async()=>({ok:true,arrayBuffer:async()=>Buffer.from('not an image')})}),/unsupported cutout/);
});
test('public designs require matching kind, safe bounded geometry and real file signature',()=>{
 const design={kind:'hat',region:'hat',name:'Blue hat',baseStyle:1,texture:png};assert(sharedDesign(design));assert.equal(sharedDesign({...design,region:'top'}),null);assert.equal(sharedDesign({...design,baseStyle:99}),null);assert.equal(sharedDesign({...design,fringe:Infinity}),null);assert.equal(sharedDesign({...design,texture:'data:image/png;base64,YmFk'}),null);
});
test('optional licensed image sources retain attribution, reject unlicensed Google results and isolate failures',async()=>{
 const env={PEXELS_API_KEY:'fixture',GOOGLE_IMAGE_SEARCH_KEY:'fixture',GOOGLE_IMAGE_SEARCH_CX:'fixture',GOOGLE_IMAGE_SEARCH_FREE_TIER_CONFIRMED:'true',UNSPLASH_ACCESS_KEY:'fixture'};assert.equal(searchProviders(env).length,3);
 const result=await searchImages('Projectile diagram',{env,fetcher:async url=>({ok:true,json:async()=>url.includes('pexels')?{photos:[{url:'https://pexels.com/photo/1',src:{large:'https://images.pexels.com/a.jpg'},photographer:'Artist',alt:'Projectile'}]}:url.includes('unsplash')?{results:[{id:'abc_123',urls:{regular:'https://images.unsplash.com/a.jpg'},links:{html:'https://unsplash.com/photos/abc_123'},user:{name:'Author'}}]}:{items:[{link:'https://example.com/unlicensed.png',image:{contextLink:'https://example.com'}},{link:'https://example.com/licensed.png',image:{contextLink:'https://example.com'},pagemap:{metatags:[{license:'https://creativecommons.org/licenses/by/4.0/'}]}}]}})});
 assert.equal(result.images.length,3);assert(result.images.every(image=>image.source&&image.credit));assert(!result.images.some(image=>image.src.includes('unlicensed')));
 assert.equal(await trackImageDownload('../bad',{env,fetcher:()=>{throw new Error('must not fetch');}}),false);
});

test('background-removal backup handles credential rejection but never shared quota or depleted credits',async()=>{
 const env={REMOVE_BG_API_KEY:'fixture-primary',REMOVE_BG_API_KEY_packup:'fixture-backup',REMOVE_BG_FREE_TIER_CONFIRMED:'true'},seen=[];
 const result=await imageOperation({action:'remove-background',image:png},{env,fetcher:async(_url,options)=>{seen.push(options.headers['X-Api-Key']);return seen.length===1?{ok:false,status:401}:{ok:true,arrayBuffer:async()=>Buffer.from(png.split(',')[1],'base64')};}});assert.equal(result.image,png);assert.deepEqual(seen,['fixture-primary','fixture-backup']);
 for(const status of [402,429]){let calls=0;await assert.rejects(imageOperation({action:'remove-background',image:png},{env,fetcher:async()=>{calls++;return{ok:false,status};}}));assert.equal(calls,1);}
});

const generationEnv={CLOUDFLARE_API_TOKEN:'fixture-image-primary',CLOUDFLARE_API_TOKEN_packup:'fixture-image-backup',CLOUDFLARE_ACCOUNT_ID:'a'.repeat(32),AI_CLOUDFLARE_STUDENT_USE_APPROVED:'true',AI_CLOUDFLARE_FREE_TIER_CONFIRMED:'true',AI_ENABLE_IMAGE_GENERATION:'true',AI_IMAGE_MODEL_APPROVED:'true'};
const generatedPng='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=';
const generatedResponse=(status=200,data={success:true,result:{image:generatedPng}})=>({status,ok:status===200,headers:{get:()=>null},json:async()=>data});
test('image generation requires separate enablement, model review, student eligibility and free-account confirmation',async()=>{
 for(const flag of ['AI_ENABLE_IMAGE_GENERATION','AI_IMAGE_MODEL_APPROVED','AI_CLOUDFLARE_STUDENT_USE_APPROVED','AI_CLOUDFLARE_FREE_TIER_CONFIRMED']){
  const env={...generationEnv,[flag]:'false'};
  assert.equal(imageCapabilities(env).imageGeneration,false);
  await assert.rejects(imageOperation({action:'generate',prompt:'Draw a plant cell'},{env,fetcher:()=>assert.fail('disabled image generation must not call a provider')}),error=>error.code==='IMAGE_NOT_CONFIGURED');
 }
});
test('images use the dedicated fixed Cloudflare model and return verified image bytes',async()=>{
 const result=await imageOperation({action:'generate',prompt:'A classroom illustration of a leaf',model:'a-paid-model'},{env:generationEnv,fetcher:async(url,options)=>{
  assert(url.endsWith('/ai/run/@cf/black-forest-labs/flux-1-schnell'));
  const body=JSON.parse(options.body);assert.equal(body.steps,4);assert(body.prompt.includes('classroom'));assert(!body.model);assert.equal(options.redirect,'error');
  return generatedResponse();
 }});
 assert.equal(result.image,'data:image/png;base64,'+generatedPng);
 assert.equal(imageCapabilities(generationEnv).imageEditing,false,'text-to-image is not sketch editing');
});
test('image credential backup is used only for rejection; rate limits and safety refusals stop immediately',async()=>{
 const seen=[];
 await imageOperation({action:'generate',prompt:'A classroom leaf'},{env:generationEnv,fetcher:async(_url,options)=>{seen.push(options.headers.Authorization);return seen.length===1?generatedResponse(401,{}):generatedResponse();}});
 assert.deepEqual(seen,['Bearer fixture-image-primary','Bearer fixture-image-backup']);
 for(const [status,data,code]of [[429,{},'IMAGE_RATE_LIMIT'],[403,{errors:[{code:'content_policy_violation'}]},'IMAGE_CONTENT_BLOCKED'],[500,{},'IMAGE_UNAVAILABLE']]){
  let calls=0;await assert.rejects(imageOperation({action:'generate',prompt:'A classroom leaf'},{env:generationEnv,fetcher:async()=>{calls++;return generatedResponse(status,data);}}),error=>error.code===code);assert.equal(calls,1);
 }
});
test('malformed image prompts and provider text cannot replace a real image',async()=>{
 for(const prompt of ['',null,'x'.repeat(1701)])await assert.rejects(imageOperation({action:'generate',prompt},{env:generationEnv,fetcher:()=>assert.fail('invalid prompt')}),error=>error.status===400);
 await assert.rejects(imageOperation({action:'generate',prompt:'A classroom leaf'},{env:generationEnv,fetcher:async()=>generatedResponse(200,{result:{image:Buffer.from('fake output').toString('base64')}})}),error=>error.code==='IMAGE_INVALID_RESPONSE');
});
