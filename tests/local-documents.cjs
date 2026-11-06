const {chromium}=require('playwright-core');
const assert=require('node:assert/strict'),path=require('node:path');
const base=process.env.CLASSIO_TEST_URL||'http://127.0.0.1:3000';
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CLASSIO_BROWSER_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 try{
  const context=await browser.newContext({viewport:{width:1366,height:900}});
  await context.request.post(base+'/api/access',{data:{action:'screen',birthDate:'2000-01-01'}});await context.request.post(base+'/api/access',{data:{action:'accept',termsAccepted:true}});
  const page=await context.newPage(),errors=[],foreignScripts=[];
  page.on('pageerror',error=>errors.push(error.message));page.on('request',request=>{if(request.resourceType()==='script'&&new URL(request.url()).origin!==base)foreignScripts.push(request.url());});
  await page.goto(base);await page.getByRole('button',{name:'Skip',exact:true}).click();await page.getByRole('button',{name:'Skip',exact:true}).waitFor({state:'hidden'});
  for(const [filename,text] of [['local-word.docx','A local Word document.'],['local-sheet.xlsx','42'],['presentation-export.pptx','Local presentation export']]){
   await page.getByRole('button',{name:/^Upload document/}).click();await page.getByRole('dialog').locator('input[type=file]').setInputFiles(path.resolve('test-results/integrated',filename));await page.getByText(filename,{exact:true}).first().click();
   await page.getByText(text,{exact:true}).first().waitFor({timeout:25000});
   console.log('PASS local browser rendering: '+filename);
   await page.getByRole('button',{name:'Dashboard',exact:true}).first().click();
  }
  // A constrained visible viewport simulates the space available above a keyboard;
  // it is not a claim that an actual iPhone keyboard was tested.
  await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'Open navigation menu',exact:true}).click();await page.getByRole('button',{name:'Your avatar',exact:true}).click();
  await page.evaluate(()=>{const root=document.documentElement;root.style.setProperty('--visible-height','360px');root.style.setProperty('--viewport-top','40px');root.dataset.shortViewport='true';});
  const dialog=page.getByRole('dialog'),box=await dialog.boundingBox();assert(box.y>=40&&box.y+box.height<=400,'Dialog must fit the constrained visual viewport');
  await page.getByRole('button',{name:'Done',exact:true}).click();
  await page.evaluate(()=>{document.documentElement.style.removeProperty('--visible-height');document.documentElement.style.removeProperty('--viewport-top');window.dispatchEvent(new Event('resize'));});
  // Exercise the fallback path used by older browsers without native dialog.
  await page.setViewportSize({width:1366,height:900});
  await page.getByRole('button',{name:'Your avatar',exact:true}).waitFor();
  await page.evaluate(()=>{HTMLDialogElement.prototype.showModal=undefined;HTMLDialogElement.prototype.close=undefined;});
  await page.getByRole('button',{name:'Your avatar',exact:true}).click();
  const close=page.getByRole('button',{name:'Close dialog',exact:true});await close.focus();await page.keyboard.press('Shift+Tab');assert(await page.getByRole('button',{name:'Done',exact:true}).evaluate(node=>node===document.activeElement),'Fallback traps focus');
  await page.keyboard.press('Escape');await dialog.waitFor({state:'hidden'});assert(await page.getByRole('button',{name:'Your avatar',exact:true}).evaluate(node=>node===document.activeElement),'Dialog restores focus');
  assert.deepEqual(foreignScripts,[]);assert.deepEqual(errors,[]);console.log('PASS constrained dialog layout, older-browser Escape/focus handling, and self-hosted document libraries');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
