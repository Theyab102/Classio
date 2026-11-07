const test=require('node:test'),assert=require('node:assert/strict');
const {imageOperation,imageCapabilities}=require('./image-service');
const {sharedDesign}=require('./avatar-sharing');
const {searchImages,searchProviders,trackImageDownload}=require('./image-search');
const png='data:image/png;base64,'+Buffer.from('89504e470d0a1a0a0000000000000000','hex').toString('base64');
test('image services require actual capabilities, and sketch bytes reach the image model',async()=>{
 assert.equal(imageCapabilities({}).imageEditing,false);
 await assert.rejects(imageOperation({action:'avatar',prompt:'blue stripes',image:png},{env:{}}),/image-generation model/);
 let sent;const result=await imageOperation({action:'avatar',prompt:'Keep my blue stripes',image:png},{env:{GEMINI_ENABLED:'true',GEMINI_API_KEY:'fixture-image-key',GEMINI_IMAGE_MODEL:'fixture-image-model'},fetcher:async(url,options)=>{sent=JSON.parse(options.body);assert.match(url,/fixture-image-model/);return{ok:true,json:async()=>({candidates:[{content:{parts:[{inlineData:{mimeType:'image/png',data:png.split(',')[1]}}]}}]})};}});
 assert.equal(sent.contents[0].parts[1].inlineData.data,png.split(',')[1]);assert.match(sent.contents[0].parts[0].text,/Preserve the sketch/);assert.equal(result.image,png);
});
test('real cutout bytes are required; errors preserve original design',async()=>{
 const env={REMOVE_BG_API_KEY:'fixture-cutout-key'};
 const result=await imageOperation({action:'remove-background',image:png},{env,fetcher:async(url,options)=>{assert.match(url,/api.remove.bg/);assert.equal(options.body.get('image_file_b64'),png.split(',')[1]);return{ok:true,arrayBuffer:async()=>Buffer.from(png.split(',')[1],'base64')};}});assert.equal(result.image,png);
 await assert.rejects(imageOperation({action:'remove-background',image:png},{env,fetcher:async()=>({ok:true,arrayBuffer:async()=>Buffer.from('not an image')})}),/unsupported cutout/);
});
test('public designs require matching kind, safe bounded geometry and real file signature',()=>{
 const design={kind:'hat',region:'hat',name:'Blue hat',baseStyle:1,texture:png};assert(sharedDesign(design));assert.equal(sharedDesign({...design,region:'top'}),null);assert.equal(sharedDesign({...design,baseStyle:99}),null);assert.equal(sharedDesign({...design,fringe:Infinity}),null);assert.equal(sharedDesign({...design,texture:'data:image/png;base64,YmFk'}),null);
});
test('optional licensed image sources retain attribution, reject unlicensed Google results and isolate failures',async()=>{
 const env={PEXELS_API_KEY:'fixture',GOOGLE_IMAGE_SEARCH_KEY:'fixture',GOOGLE_IMAGE_SEARCH_CX:'fixture',UNSPLASH_ACCESS_KEY:'fixture'};assert.equal(searchProviders(env).length,3);
 const result=await searchImages('Projectile diagram',{env,fetcher:async url=>({ok:true,json:async()=>url.includes('pexels')?{photos:[{url:'https://pexels.com/photo/1',src:{large:'https://images.pexels.com/a.jpg'},photographer:'Artist',alt:'Projectile'}]}:url.includes('unsplash')?{results:[{id:'abc_123',urls:{regular:'https://images.unsplash.com/a.jpg'},links:{html:'https://unsplash.com/photos/abc_123'},user:{name:'Author'}}]}:{items:[{link:'https://example.com/unlicensed.png',image:{contextLink:'https://example.com'}},{link:'https://example.com/licensed.png',image:{contextLink:'https://example.com'},pagemap:{metatags:[{license:'https://creativecommons.org/licenses/by/4.0/'}]}}]}})});
 assert.equal(result.images.length,3);assert(result.images.every(image=>image.source&&image.credit));assert(!result.images.some(image=>image.src.includes('unlicensed')));
 assert.equal(await trackImageDownload('../bad',{env,fetcher:()=>{throw new Error('must not fetch');}}),false);
});
