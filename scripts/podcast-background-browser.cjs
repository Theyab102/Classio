const {frontendAsset}=require('./browser-assets.cjs');
// Browser UI, navigation, IndexedDB and PCM decoding are real; provider replies
// and the slow/failing HTTP transports are explicit deterministic fixtures.
const {chromium}=require('playwright'),http=require('node:http'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../build'),output=process.env.CLASSIO_BROWSER_OUTPUT||fs.mkdtempSync(path.join(os.tmpdir(),'classio-podcast-background-'));
fs.mkdirSync(output,{recursive:true});
const server=http.createServer((req,res)=>{try{const file=frontendAsset(root,new URL(req.url,'http://local').pathname);if(!file.startsWith(root+path.sep))throw Error();res.setHeader('Content-Type',({'.js':'application/javascript','.css':'text/css','.html':'text/html','.png':'image/png','.woff2':'font/woff2'})[path.extname(file)]||'application/octet-stream');res.end(fs.readFileSync(file));}catch{res.writeHead(404);res.end();}});
const gate=()=>{let resolve;const promise=new Promise(done=>resolve=done);return{promise,resolve};};
function wav(){const n=22050,b=Buffer.alloc(44+n*2);b.write('RIFF',0);b.writeUInt32LE(36+n*2,4);b.write('WAVEfmt ',8);b.writeUInt32LE(16,16);b.writeUInt16LE(1,20);b.writeUInt16LE(1,22);b.writeUInt32LE(22050,24);b.writeUInt32LE(44100,28);b.writeUInt16LE(2,32);b.writeUInt16LE(16,34);b.write('data',36);b.writeUInt32LE(n*2,40);for(let i=0;i<n;i++)b.writeInt16LE(Math.sin(i/22050*880*Math.PI)*500,44+i*2);return b;}
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const origin='http://127.0.0.1:'+server.address().port;
 const browser=await chromium.launch({executablePath:process.env.CLASSIO_CHROMIUM_PATH||(fs.existsSync('/usr/bin/chromium')?'/usr/bin/chromium':undefined),headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});let active;
 try{for(const size of [{name:'desktop',width:1440,height:1000},{name:'tablet',width:768,height:1024},{name:'phone',width:390,height:844}]){
  if(process.env.CLASSIO_PODCAST_TARGET&&process.env.CLASSIO_PODCAST_TARGET!==size.name)continue;
  const context=await browser.newContext({viewport:size,hasTouch:size.name!=='desktop'}),page=await context.newPage();active=page;page.setDefaultTimeout(20000);
  const errors=[],script=gate(),speech=gate();let scripts=0,audios=0;
  page.on('pageerror',error=>errors.push(error.message));
  await context.addInitScript(()=>{
   localStorage.setItem('classio_access_v2',JSON.stringify({decision:'accepted',at:Date.now()}));
   const turns=Array.from({length:4},(_,i)=>({speaker:i%2?'B':'A',section:'Motion',text:'Gravity changes vertical velocity.'}));
   sessionStorage.setItem('classio_guest_workspace_v1',JSON.stringify({folders:[{id:'inbox',name:'Inbox',files:[{id:'blank_background',name:'Motion review',_isBlank:true,defaultTab:'voice',notes:'Gravity changes vertical velocity.',podcastEpisodes:[{id:'old',title:'Existing episode',turns}],studyCards:Array.from({length:8},(_,i)=>({id:'card'+i,question:'Question '+i,answer:'Answer '+i})),quiz:{questions:[{question:'Which force changes vertical velocity?',options:['Gravity','No force','Colour'],correct:0,hint:'Think downwards.',explanation:'Gravity acts downwards.'}],settings:{count:5}}}]}]}));
  });
  await context.route('**/api/status',route=>route.fulfill({json:{aiConfigured:true,providers:[{id:'groq',status:'available'}],freeOnly:true}}));
  await context.route('**/api/session',route=>route.fulfill({json:{csrf:'fixture-session',expiresAt:Date.now()+3600000}}));
  await context.route('**/*.hf.space/voices',route=>route.fulfill({json:{aria:{label:'Aria'},echo:{label:'Echo'}}}));
  await context.route('**/*.hf.space/generate-podcast',async route=>{audios++;await speech.promise;await route.fulfill({contentType:'audio/wav',body:wav()});});
  await context.route('**/api/study',async route=>{
   const input=route.request().postDataJSON();
   if(input.task==='podcast'){scripts++;await script.promise;return route.fulfill({json:{text:JSON.stringify({title:'Background episode',turns:Array.from({length:4},(_,i)=>({speaker:i%2?'B':'A',section:'Motion',text:'Gravity changes vertical velocity.'}))})}});}
   if(input.task==='flashcards')return route.fulfill({status:429,headers:{'Retry-After':'30'},json:{code:'PROVIDER_RATE_LIMIT',error:'The free AI services are busy. Try again shortly.'}});
   return route.fulfill({json:{text:'Hello. What would you like to study?'}});
  });
  const menu=async()=>{const button=page.getByRole('button',{name:'Open menu',exact:true});if(await button.isVisible())await button.click();};
  const tab=async label=>{await menu();await page.getByRole('navigation',{name:'Study tools',exact:true}).getByRole('link',{name:label,exact:true}).click();};
  const shot=async name=>{assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Page overflow: '+name);await page.screenshot({path:path.join(output,size.name+'-'+name+'.png'),fullPage:true,animations:'disabled'});};
  const saved=()=>page.evaluate(()=>JSON.parse(sessionStorage.getItem('classio_guest_workspace_v1')).folders[0].files[0]);
  await page.goto(origin);await page.getByRole('link',{name:'Open Motion review',exact:true}).click();
  const close=page.getByRole('button',{name:'Close AI Assistant',exact:true});if(await close.count())await close.click();
  await page.getByRole('button',{name:'Create podcast',exact:true}).click();await page.waitForFunction(()=>document.querySelector('.podcast-preparing'));
  await tab('Notes');assert.equal(await page.locator('.podcast-studio').count(),0,'Podcast panel did not unmount');
  const editor=page.locator('.classio-note-content[contenteditable=true]');await editor.fill('Edited while podcast generation continues.');
  await page.waitForFunction(()=>JSON.parse(sessionStorage.getItem('classio_guest_workspace_v1')).folders[0].files[0].notes.includes('Edited while'));
  script.resolve();await page.waitForFunction(()=>JSON.parse(sessionStorage.getItem('classio_guest_workspace_v1')).folders[0].files[0].podcastEpisodes.length===2);
  await tab('Quizzes');const count=page.getByRole('spinbutton',{name:'Number of questions',exact:true});await count.fill('17');await page.getByRole('button',{name:'Increase number of questions'}).click();assert.equal(await count.inputValue(),'18');await page.getByRole('button',{name:'Decrease number of questions'}).click();assert.equal(await count.inputValue(),'17');await shot('editable-question-count');
  await menu();await page.getByRole('navigation',{name:'Main menu',exact:true}).getByRole('link',{name:'Library',exact:true}).click();await page.getByRole('heading',{name:'Library',exact:true}).waitFor();
  assert(await page.getByRole('complementary',{name:'Background podcast generation'}).isVisible(),'Background status missing outside the file');
  const other=await context.newPage();await other.goto('about:blank');await other.bringToFront();speech.resolve();
  await page.waitForFunction(()=>JSON.parse(sessionStorage.getItem('classio_guest_workspace_v1')).folders[0].files[0].podcastEpisodes[0].audioKey);
  await other.close();await page.bringToFront();assert.equal(scripts,1);assert.equal(audios,4);const file=await saved();assert(file.notes.includes('Edited while'));assert.equal(file.podcastEpisodes.length,2);assert.equal(file.podcastEpisodes[1].id,'old');
  await page.getByRole('link',{name:'Open Motion review',exact:true}).click();await page.locator('.podcast-player>audio').waitFor({state:'attached'});
  const inset=await page.locator('.podcast-voice-options').evaluate(node=>{const p=node.getBoundingClientRect(),c=node.querySelector('h3').getBoundingClientRect();return{left:c.left-p.left,right:p.right-c.right,padding:parseFloat(getComputedStyle(node).paddingLeft)};});assert(inset.left>=16&&inset.padding>=16,'Voice settings touch the panel edge');await shot('padded-voice-settings');
  await tab('Flashcards');await page.getByRole('button',{name:'Generate cards',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Generate flashcards'});const cardCount=dialog.getByRole('spinbutton',{name:'Number of cards'});await cardCount.fill('50');await dialog.evaluate(async node=>{await Promise.all(node.getAnimations().map(animation=>animation.finished.catch(()=>{})));});const row=await dialog.locator('.study-count-control').evaluate(node=>({display:getComputedStyle(node).display,buttons:[...node.querySelectorAll('button')].map(button=>({height:button.getBoundingClientRect().height,top:button.getBoundingClientRect().top}))}));assert.equal(row.display,'grid','Count stepper lost its layout in the portal');assert(row.buttons.every(button=>button.height>=44),'Count stepper touch targets are too small');assert(Math.abs(row.buttons[0].top-row.buttons[1].top)<2,'Count stepper controls do not share a row');await dialog.getByRole('button',{name:'Generate cards',exact:true}).click();await dialog.getByRole('alert').filter({hasText:'Retry in 30 seconds'}).waitFor();assert(await dialog.getByRole('button',{name:/Try again in/}).isDisabled(),'Generate ignores the rate-limit deadline');await shot('editable-card-count-and-rate-limit');
  await dialog.getByRole('button',{name:'Cancel',exact:true}).click();const retry=page.getByRole('button',{name:/Try again in/});assert(await retry.isDisabled(),'Rate-limited generation can be retried too early');
  const bounds=await page.locator('.flashcard-studio>.tool-error').evaluate(node=>{const a=node.getBoundingClientRect(),b=node.parentElement.getBoundingClientRect();return{left:a.left,right:a.right,parentLeft:b.left,parentRight:b.right};});assert(bounds.left>=bounds.parentLeft-1&&bounds.right<=bounds.parentRight+1,'Error banner extends beyond the workspace');await shot('contained-rate-limit');
  assert.deepEqual(errors,[]);console.log('PASS '+size.name+': background script/audio across section, library and browser-tab navigation; concurrent note edits; existing episode preserved; saved PCM reload; count typing/steppers; padded voices; contained errors and rate-limit countdown.');await context.close();
 }console.log('Artifacts: '+output);}finally{if(active&&!active.isClosed())await active.screenshot({path:path.join(output,'last-state.png'),fullPage:true}).catch(()=>{});await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
