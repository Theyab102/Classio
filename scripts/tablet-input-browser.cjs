const {frontendAsset}=require('./browser-assets.cjs');
// Real Chromium pen, single/multi-touch, canvas pixels and Web Audio checks.
// Classification responses are fixtures; no live AI credentials are used.
const {chromium}=require('playwright'),PptxGenJS=require('pptxgenjs');
const http=require('node:http'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../build'),output=process.env.CLASSIO_BROWSER_OUTPUT||fs.mkdtempSync(path.join(os.tmpdir(),'classio-tablet-'));
fs.mkdirSync(output,{recursive:true});
const server=http.createServer((req,res)=>{try{const name=decodeURIComponent(new URL(req.url,'http://local').pathname),file=frontendAsset(root,name);if(!file.startsWith(root+path.sep))throw Error();res.setHeader('Content-Type',({'.js':'application/javascript','.mjs':'application/javascript','.html':'text/html','.css':'text/css','.woff2':'font/woff2','.png':'image/png'})[path.extname(file)]||'application/octet-stream');res.end(fs.readFileSync(file));}catch{res.writeHead(404);res.end();}});
function pdf(){
 const objects=['<< /Type /Catalog /Pages 2 0 R >>','<< /Type /Pages /Kids [3 0 R 5 0 R 7 0 R] /Count 3 >>'];
 for(let page=1;page<=3;page++){const stream=`BT /F1 28 Tf 50 1000 Td (Physics page ${page}) Tj 0 -100 Td (Forces change motion.) Tj ET`;
  objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 1100] /Resources << /Font << /F1 9 0 R >> >> /Contents ${page*2+2} 0 R >>`,`<< /Length ${Buffer.byteLength(stream)} >>\nstream\n${stream}\nendstream`);
 }objects.push('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
 let text='%PDF-1.4\n',offsets=[0];objects.forEach((body,i)=>{offsets.push(Buffer.byteLength(text));text+=`${i+1} 0 obj\n${body}\nendobj\n`;});const at=Buffer.byteLength(text);text+=`xref\n0 10\n0000000000 65535 f \n`+offsets.slice(1).map(n=>String(n).padStart(10,'0')+' 00000 n \n').join('')+`trailer\n<< /Size 10 /Root 1 0 R >>\nstartxref\n${at}\n%%EOF`;return Buffer.from(text);
}
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const origin='http://127.0.0.1:'+server.address().port;
 const browser=await chromium.launch({executablePath:process.env.CLASSIO_CHROMIUM_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});let active;
 try{
 for(const size of [{name:'desktop',width:1440,height:1000},{name:'tablet-landscape',width:1024,height:768},{name:'tablet-portrait',width:768,height:1024},{name:'phone',width:390,height:844}]){
  if(process.env.CLASSIO_TABLET_TARGET&&process.env.CLASSIO_TABLET_TARGET!==size.name)continue;
  const context=await browser.newContext({viewport:size,hasTouch:true}),page=await context.newPage();active=page;page.setDefaultTimeout(15000);const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(()=>{
   localStorage.setItem('classio_access_v2',JSON.stringify({decision:'accepted',at:Date.now()}));
   const Native=window.AudioContext;window.__soundMonitors=[];
   window.AudioContext=class extends Native{constructor(...args){super(...args);const analyser=super.createAnalyser(),silent=super.createGain();analyser.fftSize=256;silent.gain.value=0;analyser.connect(silent);silent.connect(this.destination);window.__soundMonitors.push(analyser);const create=this.createGain.bind(this);this.createGain=()=>{const gain=create();gain.connect(analyser);return gain;};}};
  });
  await page.route('**/api/session',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({csrf:'fixture-browser-csrf',expiresAt:Date.now()+3600000})}));
  await page.route('**/api/study',route=>route.fulfill({contentType:'application/json',body:'{"text":"Physics"}'}));
  await page.route('**/api/images',route=>route.fulfill({contentType:'application/json',body:'{"searchProviders":[]}'}));
  await page.goto(origin);await page.getByRole('heading',{name:'Library',exact:true}).waitFor();
  const cdp=await context.newCDPSession(page);
  async function nav(name){const menu=page.getByRole('button',{name:'Open menu',exact:true});if(await menu.isVisible())await menu.click();await page.getByRole('navigation',{name:'Study tools'}).getByRole('link',{name,exact:true}).click();}
  async function upload(name,mimeType,buffer){await page.getByRole('button',{name:'Upload document',exact:true}).click();await page.locator('input[type=file]').last().setInputFiles({name,mimeType,buffer});await page.getByRole('button',{name:'Open '+name,exact:true}).click();if(await page.getByRole('button',{name:'Close AI Assistant',exact:true}).count())await page.getByRole('button',{name:'Close AI Assistant',exact:true}).click();await nav('Source');await page.locator('.studio-ink').waitFor();await page.locator('.pdf-render-status').waitFor({state:'hidden'});await page.waitForTimeout(250);}
  async function area(selector){const node=page.locator(selector);await node.scrollIntoViewIfNeeded();return node.evaluate(el=>{const a=el.getBoundingClientRect(),c=el.querySelector('.studio-ink').getBoundingClientRect(),left=Math.max(a.left,c.left,0),right=Math.min(a.right,c.right,innerWidth),top=Math.max(a.top,c.top,0),bottom=Math.min(a.bottom,c.bottom,innerHeight);return{x:(left+right)/2,y:(top+bottom)/2,w:right-left,h:bottom-top};});}
  async function swipe(box,dx,dy,count=1,expectAnimation=true){
   const points=Array.from({length:count},(_,i)=>({x:box.x+i*25,y:box.y+i*15,id:i+1}));
   const shouldTurn=expectAnimation&&Math.abs(dx)>55&&Math.abs(dx)>Math.abs(dy)*1.5;
   await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:points});
   for(let n=1;n<=10;n++)await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:points.map(p=>({...p,x:p.x+dx*n/10,y:p.y+dy*n/10}))});
   if(shouldTurn){
    await page.locator('[data-page-turn="dragging"]').waitFor();await page.waitForFunction(()=>document.querySelector('.page-turn-incoming')?.children.length>0);
    const position=await page.evaluate(()=>{const x=selector=>new DOMMatrixReadOnly(getComputedStyle(document.querySelector(selector)).transform).m41;return{out:x('.page-turn-outgoing'),incoming:x('.page-turn-incoming'),width:document.querySelector('[data-page-turn]').clientWidth,inert:document.querySelector('.page-turn-overlay').inert};});
    assert(Math.abs(position.out-dx)<3,'Current page did not follow the finger');assert(Math.sign(position.incoming)===-Math.sign(dx),'Adjacent page entered from the wrong edge');assert(Math.abs(Math.abs(position.incoming-position.out)-position.width)<1,'Both pages do not travel together');assert(position.inert,'Temporary page copies are interactive');
    const kind=await page.locator('[data-page-turn]').evaluate(n=>n.classList.contains('pdf-scroll')?'pdf':'pptx');await page.screenshot({path:path.join(output,size.name+'-'+kind+'-'+(dx<0?'left':'right')+'-'+Math.round(Math.abs(dx))+'-drag.png')});
   }
   await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
   if(shouldTurn){await page.waitForFunction(()=>document.querySelector('[data-page-turn="turning"]'));const ends=await page.locator('.page-turn-plane').evaluateAll(nodes=>nodes.map(n=>n.getAnimations()[0]?.effect.getKeyframes().at(-1).transform));assert(ends[0].includes(String(Math.sign(dx)*await page.locator('[data-page-turn]').evaluate(n=>n.clientWidth))),'Outgoing animation finished on the wrong side');assert(/^translate3d\(0px,\s*0px,\s*0px\)$/.test(ends[1]),'Incoming page did not settle at the reading position');}
   await page.locator('.page-turn-overlay').waitFor({state:'detached'});await page.waitForTimeout(30);
  }

  async function pen(box){await cdp.send('Input.dispatchMouseEvent',{type:'mouseMoved',x:box.x-30,y:box.y-20,pointerType:'pen'});await cdp.send('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,x:box.x-30,y:box.y-20,pointerType:'pen'});for(let n=1;n<=6;n++)await cdp.send('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,x:box.x-30+n*10,y:box.y-20+n*6,pointerType:'pen'});await cdp.send('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,x:box.x+30,y:box.y+16,pointerType:'pen'});}
  // Inspect a bitmap copy: repeated getImageData on the live ink canvas can
  // switch Chromium's GPU backing to CPU in the middle of the interaction.
  // Pixel assertions still verify the actual rendered output of the app.
  const expectPixels=empty=>page.waitForFunction(async empty=>{const c=document.querySelector('.studio-ink');if(!c)return false;const bitmap=await createImageBitmap(c),copy=new OffscreenCanvas(c.width,c.height),context=copy.getContext('2d',{willReadFrequently:true});context.drawImage(bitmap,0,0);bitmap.close();const data=context.getImageData(0,0,copy.width,copy.height).data;let n=0;for(let i=3;i<data.length;i+=4)if(data[i])n++;return empty?n===0:n>20;},empty);
  const ink=page.getByRole('toolbar',{name:'Annotation tools'}),undo=page.getByRole('button',{name:'Undo ink',exact:true}),redo=page.getByRole('button',{name:'Redo ink',exact:true});
  await upload('Tablet physics.pdf','application/pdf',pdf());
  assert(await undo.isVisible());assert(await undo.isDisabled());assert(await redo.isVisible());
  const storedInk=(property,name)=>page.evaluate(({property,name})=>JSON.stringify(JSON.parse(sessionStorage.getItem('classio_guest_workspace_v1')).folders.flatMap(f=>f.files).find(f=>f.name===name)?.[property]||{}),{property,name});
  await ink.getByRole('button',{name:'Pen',exact:true}).click();let box=await area('.pdf-scroll');await pen(box);await page.waitForFunction(()=>!document.querySelector('[aria-label="Undo ink"]').disabled);await expectPixels(false);const savedStrokes=await storedInk('pdfInk','Tablet physics.pdf');await page.locator('.studio-ink').evaluate(c=>c.style.transform='scale(0)');await undo.click();await expectPixels(true);await page.locator('.studio-ink').evaluate(c=>c.style.transform='');await redo.click();await expectPixels(false);assert.equal(await storedInk('pdfInk','Tablet physics.pdf'),savedStrokes,'Redo changed the actual saved stroke');await undo.click();await expectPixels(true);await redo.click();await expectPixels(false);assert.equal(await storedInk('pdfInk','Tablet physics.pdf'),savedStrokes);
  box=await area('.pdf-scroll');await swipe(box,-30,0,1);assert.equal(await page.getByRole('spinbutton',{name:'PDF page number'}).inputValue(),'1','A short swipe changed the page');box=await area('.pdf-scroll');await swipe(box,95,0,2,false);assert.equal(await page.getByRole('spinbutton',{name:'PDF page number'}).inputValue(),'1','Boundary swipe left the first page');assert.equal(await storedInk('pdfInk','Tablet physics.pdf'),savedStrokes);
  // Fingers scroll the PDF while Pen remains selected, without changing pixels.
  box=await area('.pdf-scroll');await swipe(box,0,-90,2);assert(await page.locator('.pdf-scroll').evaluate(n=>n.scrollTop)>50,'Two fingers did not scroll the PDF');assert.equal(await storedInk('pdfInk','Tablet physics.pdf'),savedStrokes,'Two-finger scroll painted ink');assert.equal(await page.getByRole('spinbutton',{name:'PDF page number'}).inputValue(),'1');
  box=await area('.pdf-scroll');await swipe(box,0,90,2);assert(await page.locator('.pdf-scroll').evaluate(n=>n.scrollTop)<10,'Two fingers did not scroll back up');box=await area('.pdf-scroll');await swipe(box,0,-90,1);assert(await page.locator('.pdf-scroll').evaluate(n=>n.scrollTop)>50,'One finger did not scroll with Pen selected');box=await area('.pdf-scroll');await swipe(box,-95,0,2);await page.waitForFunction(()=>document.querySelector('[aria-label="PDF page number"]').value==='2');await expectPixels(true);assert(await undo.isDisabled(),'A page swipe was saved as ink');
  box=await area('.pdf-scroll');await swipe(box,95,0,1);await page.waitForFunction(()=>document.querySelector('[aria-label="PDF page number"]').value==='1');await expectPixels(false);assert.equal(await storedInk('pdfInk','Tablet physics.pdf'),savedStrokes,'Turning pages lost existing ink');
  // Cancellation discards an in-progress pen stroke rather than saving a line.
  const surface=page.locator('.studio-ink');await surface.dispatchEvent('pointerdown',{pointerId:99,pointerType:'touch',button:0,clientX:box.x,clientY:box.y});await surface.dispatchEvent('pointercancel',{pointerId:99,pointerType:'touch'});assert.equal(await storedInk('pdfInk','Tablet physics.pdf'),savedStrokes);
  await ink.getByRole('button',{name:'Scroll',exact:true}).click();assert(await undo.isVisible(),'Undo disappeared in scroll mode');
  box=await area('.pdf-scroll');await swipe(box,-95,0,1);await page.waitForFunction(()=>document.querySelector('[aria-label="PDF page number"]').value==='2');
  await page.screenshot({path:path.join(output,size.name+'-pdf.png'),fullPage:true});
  await page.getByRole('button',{name:'Back to library',exact:true}).click();
  const deck=new PptxGenJS();deck.layout='LAYOUT_WIDE';for(let i=1;i<=3;i++)deck.addSlide().addText('Tablet slide '+i,{x:1,y:1,w:10,h:2,fontSize:30});
  await upload('Tablet physics.pptx','application/vnd.openxmlformats-officedocument.presentationml.presentation',Buffer.from(await deck.write({outputType:'nodebuffer'})));
  await ink.getByRole('button',{name:'Pen',exact:true}).click();assert.equal(await ink.evaluate(n=>!!n.closest('.presentation-source-size')),false,'The drawing controls shrink with the slide');assert((await ink.boundingBox()).width>(await page.locator('.presentation-source-scroll').boundingBox()).width*.8,'The drawing toolbar is narrower than its viewport');box=await area('.presentation-source-scroll');await pen(box);await page.waitForFunction(()=>!document.querySelector('[aria-label="Undo ink"]').disabled);await expectPixels(false);const slideInk=await storedInk('powerPointInk','Tablet physics.pptx');
  box=await area('.presentation-source-scroll');await swipe(box,-95,0,2);await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='2');await expectPixels(true);
  box=await area('.presentation-source-scroll');await swipe(box,95,0,1);await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='1');await expectPixels(false);assert.equal(await storedInk('powerPointInk','Tablet physics.pptx'),slideInk);
  // Finger ink remains available explicitly; adding a second finger cancels the
  // provisional stroke before navigating, rather than committing either finger.
  await page.getByLabel('Draw with finger',{exact:true}).check();box=await area('.presentation-source-scroll');await swipe(box,-95,0,2);await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='2');await expectPixels(true);assert.equal(await storedInk('powerPointInk','Tablet physics.pptx'),slideInk,'Gesture committed provisional finger ink');box=await area('.presentation-source-scroll');await swipe(box,35,10,1);await expectPixels(false);await undo.click();await expectPixels(true);await page.getByLabel('Draw with finger',{exact:true}).uncheck();
  box=await area('.presentation-source-scroll');await swipe({...box,x:box.x+box.w*.2},-Math.round(box.w*.45),0,2);await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='3');box=await area('.presentation-source-scroll');await swipe(box,-95,0,1,false);assert.equal(await page.getByRole('spinbutton',{name:'Slide number'}).inputValue(),'3','Boundary swipe left the last slide');box=await area('.presentation-source-scroll');await swipe(box,95,0,1);await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='2');
  await page.emulateMedia({reducedMotion:'reduce'});box=await area('.presentation-source-scroll');await swipe(box,-95,0,2,false);await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='3');assert.equal(await page.locator('.page-turn-overlay').count(),0,'Reduced motion still moved page copies');await page.getByRole('button',{name:'Previous slide',exact:true}).click();await page.waitForFunction(()=>document.querySelector('[aria-label="Slide number"]').value==='2');await page.emulateMedia({reducedMotion:'no-preference'});
  await page.screenshot({path:path.join(output,size.name+'-slides.png'),fullPage:true});await page.evaluate(()=>{localStorage.setItem('classio_theme','dark');window.dispatchEvent(new StorageEvent('storage',{key:'classio_theme'}));});await page.waitForFunction(()=>document.documentElement.dataset.theme==='dark');assert(await undo.isVisible());assert(await redo.isVisible());await page.screenshot({path:path.join(output,size.name+'-dark-tools.png'),fullPage:true});
  // Exercise real synthesis and measure its output, including the off switch.
  {const menu=page.getByRole('button',{name:'Open menu',exact:true});if(await menu.isVisible())await menu.click();await page.getByRole('link',{name:'Settings',exact:true}).click();}const toggle=page.getByRole('button',{name:'Study sound effects',exact:true});await toggle.click();await page.waitForFunction(()=>window.__soundMonitors.some(a=>a.context.state==='running'));
  // Meter on the actual audio rendering thread, so a busy UI frame cannot
  // miss a short envelope peak. The worklet emits silence and changes no sound.
  const peak=await page.evaluate(async()=>{
   const monitor=window.__soundMonitors.at(-1),context=monitor.context;
   const url=URL.createObjectURL(new Blob([`registerProcessor('classio-test-meter',class extends AudioWorkletProcessor{constructor(){super();this.peak=0;this.port.onmessage=()=>this.port.postMessage(this.peak);}process(inputs){for(const channel of inputs[0]||[])for(const sample of channel)this.peak=Math.max(this.peak,Math.abs(sample));return true;}});`],{type:'application/javascript'}));
   try{await context.audioWorklet.addModule(url);}finally{URL.revokeObjectURL(url);}
   const meter=new AudioWorkletNode(context,'classio-test-meter');monitor.connect(meter);meter.connect(context.destination);
   document.querySelector('[aria-label="Study sound effects"]').click();await new Promise(resolve=>setTimeout(resolve,500));
   const peak=await new Promise(resolve=>{meter.port.onmessage=event=>resolve(event.data);meter.port.postMessage('peak');});monitor.disconnect(meter);meter.disconnect();meter.port.close();return peak;
  });assert(peak>.12&&peak<.8,'Study sound is not audibly boosted or may clip: '+peak);

  await toggle.click();assert.equal(await toggle.getAttribute('aria-pressed'),'false');
  assert.deepEqual(errors,[],'Uncaught browser errors');assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Page overflow');
  console.log('PASS '+size.name+': real pen, visible undo/redo, finger-tracked PDF/slide motion in both directions, real adjacent previews, short/boundary rollback, reduced motion, PDF scrolling, no gesture ink, retained page ink, finger-ink override and native sound (peak '+peak.toFixed(3)+').');await cdp.detach();await context.close();
 }
 console.log('Artifacts: '+output);
 }catch(error){if(active)await active.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw error;}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
