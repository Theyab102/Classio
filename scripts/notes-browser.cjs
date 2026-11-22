const {frontendAsset}=require('./browser-assets.cjs');
// Real editor interactions and responsive layouts. AI transports are fixtures,
// including OUTPUT_LIMIT, rate limits and cancellation; no live AI claim.
const {chromium}=require('playwright');
const assert=require('node:assert/strict'),http=require('node:http'),fs=require('node:fs'),path=require('node:path'),os=require('node:os');
const root=path.resolve(__dirname,'../build'),output=process.env.CLASSIO_NOTES_OUTPUT||fs.mkdtempSync(path.join(os.tmpdir(),'classio-notes-'));
fs.mkdirSync(output,{recursive:true});
const mime={'.html':'text/html','.js':'application/javascript','.css':'text/css','.png':'image/png','.woff2':'font/woff2','.json':'application/json'};
const server=http.createServer((req,res)=>{const name=new URL(req.url,'http://localhost').pathname,target=frontendAsset(root,name);if(!target.startsWith(root+path.sep)){res.writeHead(403);return res.end();}try{res.setHeader('Content-Type',mime[path.extname(target)]||'application/octet-stream');res.end(fs.readFileSync(target));}catch{res.writeHead(404);res.end();}});
(async()=>{
 if(!process.env.CLASSIO_NOTE_TEST_ORIGIN)await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const origin=process.env.CLASSIO_NOTE_TEST_ORIGIN||'http://127.0.0.1:'+server.address().port;
 const browser=await chromium.launch({executablePath:process.env.CLASSIO_CHROMIUM_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader']});let active;
 try{
 for(const size of [{name:'desktop',width:1440,height:1000},{name:'tablet-landscape',width:1024,height:768},{name:'tablet-portrait',width:768,height:1024},{name:'phone',width:390,height:844}]){
  if(process.env.CLASSIO_NOTES_TARGET&&process.env.CLASSIO_NOTES_TARGET!==size.name)continue;
  const context=await browser.newContext({viewport:size,hasTouch:size.name!=='desktop'}),page=await context.newPage();active=page;page.setDefaultTimeout(20000);
  const errors=[],calls=[];let mode='success',truncated=false,releaseLate,lateRequested;
  page.on('pageerror',error=>errors.push(error.message));
  await page.addInitScript(()=>localStorage.setItem('classio_access_v2',JSON.stringify({decision:'accepted',at:Date.now()})));
  await page.route('**/api/session',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({csrf:'fixture-browser-csrf',expiresAt:Date.now()+3600000})}));
  await page.route('**/api/study',async route=>{
   const data=route.request().postDataJSON(),prompt=data.messages[0].content;calls.push(data);
   if(prompt.includes('compact course outline')){
    if(mode==='cancel'){await new Promise(resolve=>{releaseLate=resolve;lateRequested?.();});return route.fulfill({contentType:'application/json',body:JSON.stringify({text:JSON.stringify(Array.from({length:4},(_,i)=>({title:'Late '+i,focus:'Late source'})))})}).catch(()=>{});}
    return route.fulfill({contentType:'application/json',body:JSON.stringify({text:JSON.stringify(Array.from({length:4},(_,i)=>({title:'Source section '+(i+1),focus:'Explain the assigned source section.'})))})});
   }
   if(prompt.includes('one complete lesson')){
    const index=Number(data.messages[1].content.match(/Lesson (\d+) of/)?.[1]);
    if(index===2&&mode==='fail')return route.fulfill({status:429,contentType:'application/json',body:'{"error":"Provider limit: wait before trying again.","code":"RATE_LIMIT"}'});
    if(index===2&&!truncated){truncated=true;return route.fulfill({status:503,contentType:'application/json',body:'{"error":"The answer was too long to finish.","code":"OUTPUT_LIMIT"}'});}
    return route.fulfill({contentType:'application/json',body:JSON.stringify({text:JSON.stringify([{title:'Source section '+index,lesson:'## شرح الدرس\n\nشرح يستند إلى المصدر مع مثال وخلاصة.',question:'ما الإجابة الصحيحة؟',options:['أ','ب','ج'],correct:0,explanation:'الإجابة الأولى مستندة إلى المصدر.'}])})});
   }
   return route.fulfill({contentType:'application/json',body:JSON.stringify({text:'| Quantity | Meaning |\n| --- | --- |\n| Velocity | Motion over time |'})});
  });
  await page.goto(origin);await page.getByRole('button',{name:'Blank document',exact:true}).click();
  if(await page.getByRole('button',{name:'Close AI Assistant',exact:true}).count())await page.getByRole('button',{name:'Close AI Assistant',exact:true}).click();
  const editor=page.getByRole('textbox',{name:'Study notes',exact:true});
  const choose=async(label,value)=>{await page.getByRole('combobox',{name:label,exact:true}).click();await page.getByRole('option',{name:value,exact:label!=='Notes mode'}).click();};
  const appearance=async value=>{const open=page.getByRole('button',{name:'Open menu',exact:true});if(await open.isVisible())await open.click();if(!await page.getByRole('combobox',{name:'Appearance',exact:true}).count())await page.getByRole('button',{name:'Expand sidebar',exact:true}).click();await choose('Appearance',value);const close=page.getByRole('button',{name:'Close menu',exact:true});if(await close.isVisible())await close.click();};
  const shot=async name=>{assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),size.name+' page overflow at '+name);await page.screenshot({path:path.join(output,size.name+'-'+name+'.png'),fullPage:true,animations:'disabled'});};
  const tableOption=async name=>{await page.getByRole('button',{name:'Table options',exact:true}).click();await page.getByRole('menuitem',{name,exact:true}).click();};
  await editor.fill('Projectile Motion');await page.getByRole('button',{name:'More formatting',exact:true}).click();await choose('Paragraph style','Title');
  // Set a real DOM text selection in the middle of a heading, like a caret click.
  await editor.locator('h1').evaluate(node=>{const selection=getSelection();selection.collapse(node.firstChild,6);});await page.keyboard.press('ArrowRight');
  await page.getByRole('button',{name:'Insert table',exact:true}).click();
  assert.equal(await editor.locator('h1').innerText(),'Projectile Motion','Inserting a table split the heading');
  assert.equal(await editor.locator('th').count(),0,'Ordinary table cells were made bold headers');
  assert(await editor.locator('td p').evaluateAll(nodes=>nodes.every(node=>getComputedStyle(node,'::before').content==='none')),'Empty table cells display the note placeholder');
  let table=editor.locator('table').first(),first=table.locator('td p').first();
  await first.click();await page.keyboard.type('Velocity');await page.keyboard.press('Home');await page.keyboard.press('Shift+End');
  await page.getByRole('button',{name:'Bold',exact:true}).click();await page.getByRole('button',{name:'Italic',exact:true}).click();await choose('Font family','Serif');await choose('Font size','24');await choose('Align text','Center');
  const styled=first.locator('em').first();assert(await styled.evaluate(node=>{const style=getComputedStyle(node);return style.fontFamily.includes('Fraunces')&&style.fontSize==='24px'&&Number(style.fontWeight)>=700&&style.fontStyle==='italic';}),'Table cell formatting was not applied');assert.equal(await first.evaluate(node=>getComputedStyle(node).textAlign),'center');
  await page.getByRole('button',{name:'Add row',exact:true}).click();assert.equal(await table.locator('tr').count(),4);await page.getByRole('button',{name:'Undo',exact:true}).click();assert.equal(await table.locator('tr').count(),3);await page.getByRole('button',{name:'Redo',exact:true}).click();assert.equal(await table.locator('tr').count(),4);
  await tableOption('Toggle header row');assert.equal(await table.locator('th').count(),3);await tableOption('Toggle header row');assert.equal(await table.locator('th').count(),0);
  await first.click();await page.keyboard.press('ArrowLeft');
  if(size.name==='desktop'){
    await page.keyboard.down('Shift');await table.locator('tr').first().locator('td p').nth(1).click();await page.keyboard.up('Shift');await tableOption('Merge cells');
  }else await tableOption('Merge with next cell');
  assert.equal(await table.locator('tr').first().locator('td').count(),2);await tableOption('Split cell');assert.equal(await table.locator('tr').first().locator('td').count(),3);
  console.log(size.name+': table formatting and undo passed');
  await table.locator('tr').nth(1).locator('td p').first().click();await page.keyboard.insertText('A detailed explanation grows vertically inside this table. '.repeat(12));
  assert(await table.locator('td').evaluateAll(nodes=>nodes.every(node=>node.scrollHeight<=node.clientHeight+2)),'A table cell clips text or scrolls vertically');
  const cell=table.locator('tr').first().locator('td').first();await cell.scrollIntoViewIfNeeded();const box=await cell.boundingBox();await page.mouse.move(box.x+box.width-1,box.y+box.height/2);
  assert.equal(await cell.evaluate(node=>getComputedStyle(node).cursor),'col-resize','Column boundary lacks a resize cursor');
  await page.mouse.down();await page.mouse.move(Math.min(size.width-3,box.x+box.width+400),box.y+box.height/2,{steps:10});await page.mouse.up();
  assert(await table.locator('td[colwidth]').count()>0,'Column drag did not persist widths');
  assert(await table.evaluate(node=>node.getBoundingClientRect().width<=node.closest('.classio-note-content').clientWidth+1),'Table overflowed after resizing');
  assert(await table.locator('tr').first().locator('td').evaluateAll(nodes=>nodes.every(node=>node.getBoundingClientRect().width>=70)),'Resize created an unusably thin column');
  await shot('table-light');await appearance('Dark');await shot('table-dark');await appearance('Light');console.log(size.name+': table resize and wrapping passed');
  // Save/restore through actual guest navigation, preserving rich marks and widths.
  await page.getByRole('button',{name:'Back to library',exact:true}).click();await page.getByRole('link',{name:'Open Untitled note',exact:true}).click();
  table=editor.locator('table').first();first=table.locator('td p').first();assert.equal(await first.innerText(),'Velocity');assert(await first.locator('span').evaluateAll(nodes=>nodes.some(node=>getComputedStyle(node).fontSize==='24px'&&getComputedStyle(node).fontFamily.includes('Fraunces'))),'Formatting was lost on restoration');assert(await table.locator('td[colwidth]').count()>0,'Widths were lost on restoration');
  for(const key of ['Backspace','Delete']){await first.click();await page.getByRole('button',{name:'Select table',exact:true}).click();await editor.locator('.tableWrapper.ProseMirror-selectednode').waitFor();await editor.press(key);assert.equal(await editor.locator('table').count(),0,key+' left the selected table behind');assert.equal(await editor.locator('h1').innerText(),'Projectile Motion');await editor.press('Control+z');assert.equal(await editor.locator('table').count(),1,'Undo did not restore the table');}
  await first.click();await page.keyboard.press('Home');await page.keyboard.press('Shift+End');await page.waitForFunction(()=>{const e=document.querySelector('.classio-note-content').editor;return e.state.doc.textBetween(e.state.selection.from,e.state.selection.to)==='Velocity';});await page.keyboard.press('Backspace');assert.equal((await first.innerText()).trim(),'');await editor.press('Control+z');assert.equal(await first.innerText(),'Velocity');
  await first.click();await page.getByRole('button',{name:'Insert table',exact:true}).click();assert.equal(await editor.locator('table table').count(),0,'Repeated insertion created a nested table');assert.equal(await editor.locator('table').count(),2);
  await page.getByRole('button',{name:'Make a table with AI',exact:true}).click();await page.getByText('Table inserted below the current block. You can edit every cell.',{exact:true}).waitFor();assert.equal(await editor.locator('table table').count(),0,'AI table was nested');assert.equal(await editor.locator('table').count(),3);
  await page.getByRole('button',{name:'Generate',exact:true}).click();const toggle=page.getByRole('button',{name:/Include relevant images/});assert.equal(await toggle.getAttribute('aria-pressed'),'false');await toggle.click();assert.equal(await toggle.getAttribute('aria-pressed'),'true');await shot('generation-toggle');await page.getByRole('button',{name:'Close Generate notes',exact:true}).click();
  console.log(size.name+': table restoration and AI insertion passed');
  // An older desktop table must remain readable after rotating/reopening on a phone.
  await editor.fill('Saved wide table');await editor.press('Control+End');await editor.press('Enter');
  await editor.evaluate(node=>{const transfer=new DataTransfer();transfer.setData('text/html','<table><tr>'+Array.from({length:8},(_,i)=>'<th><p>Heading '+(i+1)+'</p></th>').join('')+'</tr><tr>'+Array.from({length:8},(_,i)=>'<td><p>Source evidence '+(i+1)+'</p></td>').join('')+'</tr></table>');node.dispatchEvent(new ClipboardEvent('paste',{bubbles:true,clipboardData:transfer}));});
  await page.setViewportSize({width:390,height:844});await page.waitForFunction(()=>document.querySelector('.tableWrapper[data-stacked=true] td'));
  assert.equal(await editor.locator('td').count(),8);assert((await editor.innerText()).includes('Source evidence 8'));
  assert(await editor.locator('td').evaluateAll(nodes=>nodes.every(node=>node.getBoundingClientRect().width>=72&&node.scrollWidth<=node.clientWidth+2)),'Older wide table became cramped or overflowed');
  await shot('saved-wide-table-phone');await page.setViewportSize({width:size.width,height:size.height});
  // Legacy saved HTML enters the same paste/sanitize path as restored notes.
  await editor.fill('');await editor.evaluate(node=>{const transfer=new DataTransfer();transfer.setData('text/html','<h2>Projectile motion</h2><p><strong>Keep the time-of-flight explanation.</strong></p><img src="https://example.com/forest.jpg" alt="Bomb Crater in King’s Wood – geograph.org.uk"><p><a href="https://commons.wikimedia.org/forest">David Anstiss · CC BY-SA 2.0</a></p><img src="https://example.com/manual.png" alt="My manual sketch" data-classio-image-origin="manual">');node.dispatchEvent(new ClipboardEvent('paste',{bubbles:true,clipboardData:transfer}));});
  await page.getByRole('button',{name:'Insert image',exact:true}).click();await page.getByText('This saved image may not explain the surrounding section. Review it before keeping it.',{exact:true}).waitFor();await shot('image-review');await page.getByRole('button',{name:'Remove image 1',exact:true}).click();assert.equal(await editor.locator('img').count(),1);assert(await editor.innerHTML().then(html=>html.includes('manual.png')&&html.includes('<strong>Keep the time-of-flight explanation.')),'Image review damaged manual images or writing');await page.getByRole('button',{name:'Close Images',exact:true}).click();
  await editor.fill(Array.from({length:4},(_,i)=>`SOURCE${i+1}: تتناول هذه الوحدة الحركة والقوة وتشرح المفاهيم مع أمثلة واضحة. ${'دليل من المصدر '.repeat(50)}\n\n`).join(''));
  await choose('Notes mode','Guided lessons with questions');await choose('Number of lessons','4');await page.getByRole('button',{name:'Generate lessons',exact:true}).click();await page.locator('.lesson-outline button').nth(3).waitFor();assert(truncated,'Truncation fixture did not execute');assert(calls.filter(call=>call.messages[0].content.includes('one complete lesson')).some(call=>call.maxTokens===1400),'No shorter retry after truncation');
  await page.locator('.answer-choices button').first().click();await page.getByText('1 of 4 lessons completed',{exact:true}).waitFor();
  const last=calls.filter(call=>call.messages[0].content.includes('one complete lesson')).at(-1);assert(last.messages[1].content.split('Assigned source section')[1].includes('SOURCE4'),'Later source section was not used');assert(last.messages[0].content.includes('in Arabic'));
  mode='fail';await page.getByRole('button',{name:'New course',exact:true}).click();await page.getByRole('button',{name:'Generate new course',exact:true}).click();await page.getByRole('alert').filter({hasText:'Provider limit'}).waitFor();assert.equal(await page.locator('.lesson-outline button').count(),4);await page.getByText('1 of 4 lessons completed',{exact:true}).waitFor();
  mode='cancel';const blockedRequest=new Promise(resolve=>lateRequested=resolve);await page.getByRole('button',{name:'New course',exact:true}).click();await page.getByRole('button',{name:'Generate new course',exact:true}).click();await blockedRequest;await page.getByRole('button',{name:'Cancel generation',exact:true}).click();releaseLate();await page.waitForTimeout(100);assert.equal(await page.getByText('Late 0',{exact:true}).count(),0);await page.getByText('1 of 4 lessons completed',{exact:true}).waitFor();
  await shot('lessons-light');
  // Source form controls share a baseline and touch size.
  const fields=await page.locator('.course-generator input,.course-generator textarea,.course-generator .choice-trigger,.course-generator>.quiet-button').evaluateAll(nodes=>nodes.map(node=>({height:node.getBoundingClientRect().height,bottom:node.getBoundingClientRect().bottom})));assert(fields.every(field=>field.height>=44),'Small course control');if(size.width>=1200)assert(Math.max(...fields.map(field=>field.bottom))-Math.min(...fields.map(field=>field.bottom))<2,'Course form baseline mismatch');
  await appearance('Dark');await shot('lessons-dark');
  assert.deepEqual(errors,[],'Uncaught runtime errors');console.log('PASS '+size.name+': real table insertion/formatting/undo/resize/restoration, image review/toggle, Arabic section coverage, bounded truncation retry, course preservation and cancellation. AI responses are fixtures.');await context.close();
 }
 console.log('Artifacts: '+output);
 }catch(error){if(active)await active.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});throw error;}finally{await browser.close();if(server.listening)await new Promise(resolve=>server.close(resolve));}
})().catch(error=>{console.error(error);process.exitCode=1;});
