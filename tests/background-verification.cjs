const {chromium}=require('playwright-core'),AxeBuilder=require('@axe-core/playwright').default;
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const base=process.env.CLASSIO_TEST_URL||'http://127.0.0.1:3000',out=path.resolve('test-results/background');fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CLASSIO_BROWSER_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 try{
  const context=await browser.newContext({viewport:{width:390,height:844},colorScheme:'dark',reducedMotion:'reduce'});
  await context.addInitScript(()=>localStorage.setItem('classio_dark','true'));
  const page=await context.newPage(),errors=[];page.on('pageerror',error=>errors.push(error.message));
  let release,hold=true;
  const gate=new Promise(resolve=>{release=()=>{hold=false;resolve();};});
  await page.route('**/api/access',async route=>{if(hold&&route.request().method()==='GET')await gate;await route.continue().catch(()=>{});});
  await page.goto(base);await page.getByRole('heading',{name:'Your space to learn'}).waitFor();
  assert.equal(await page.getByText('Checking your saved verification…',{exact:true}).count(),0);
  assert.equal(await page.getByLabel('Date of birth',{exact:true}).count(),0);assert(await page.getByRole('button',{name:'Get started',exact:true}).isDisabled());
  await page.screenshot({path:path.join(out,'background-welcome-iphone-size.png')});release();
  await page.getByRole('button',{name:'Get started',exact:true}).click();await page.getByLabel('Date of birth',{exact:true}).fill('2000-01-01');await page.getByRole('button',{name:'Continue',exact:true}).click();await page.getByRole('checkbox').check();await page.getByRole('button',{name:'Open workspace',exact:true}).click();await page.getByRole('heading',{name:'Dashboard',exact:true}).waitFor();
  const cookie=(await context.cookies()).find(item=>item.name==='classio_access_v2');assert(cookie.httpOnly&&cookie.expires>Date.now()/1000+300*86400);
  console.log('PASS immediate welcome while checking, no checking screen or premature age form, real screening/terms/cookie confirmation');
  await page.unroute('**/api/access');let fail=true;
  await page.route('**/api/access',async route=>{if(fail)await route.abort('internetdisconnected');else await route.continue();});
  await page.reload();await page.getByRole('alert').waitFor();assert.equal(await page.getByLabel('Date of birth',{exact:true}).count(),0);assert.equal(await page.getByRole('heading',{name:'Dashboard',exact:true}).count(),0);
  const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa','wcag22aa']).analyze();fs.writeFileSync(path.join(out,'a11y-entry-dark.json'),JSON.stringify(result.violations,null,2));assert.deepEqual(result.violations.map(item=>({id:item.id,nodes:item.nodes.map(node=>node.target)})),[]);
  await page.screenshot({path:path.join(out,'recoverable-access-dark.png')});fail=false;await page.evaluate(()=>window.dispatchEvent(new Event('online')));await page.getByRole('heading',{name:'Dashboard',exact:true}).waitFor();assert.equal(await page.getByLabel('Date of birth',{exact:true}).count(),0);
  console.log('PASS offline failure preserves verification, blocks access truthfully and automatically restores on connectivity return; dark-mode automated accessibility');
  const skip=page.getByRole('button',{name:'Skip',exact:true});if(await skip.count())await skip.click();
  await page.getByRole('button',{name:'Open navigation menu',exact:true}).click();await page.getByRole('button',{name:'Your avatar',exact:true}).click();
  await page.getByRole('button',{name:'Village',exact:true}).click();
  for(const species of ['Human','Cat','Bear','Rabbit','Fox','Deer']){await page.getByRole('button',{name:species,exact:true}).click();assert.equal(await page.getByRole('button',{name:species,exact:true}).getAttribute('aria-pressed'),'true');}
  await page.getByRole('button',{name:'Fox',exact:true}).click();await page.setViewportSize({width:1366,height:900});
  const avatars=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21aa','wcag22aa']).analyze();fs.writeFileSync(path.join(out,'a11y-village.json'),JSON.stringify(avatars.violations,null,2));assert.deepEqual(avatars.violations.map(item=>({id:item.id,nodes:item.nodes.map(node=>node.target)})),[]);
  await page.screenshot({path:path.join(out,'village-character-fox.png')});
  for(const size of [{width:320,height:740},{width:390,height:844},{width:844,height:390},{width:768,height:1024},{width:1024,height:768}]){await page.setViewportSize(size);const box=await page.getByRole('dialog').boundingBox();assert(box.x>=-1&&box.y>=-1&&box.x+box.width<=size.width+1&&box.y+box.height<=size.height+1);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
  await page.getByRole('button',{name:'Done',exact:true}).click();assert.deepEqual(errors,[]);console.log('PASS all village species/customization and five phone/tablet viewport fits, no page errors, sampled character accessibility');
  await context.close();
  const discarded=await browser.newContext({viewport:{width:390,height:844}}),lost=await discarded.newPage();
  await lost.route('**/api/access',async route=>{if(route.request().method()!=='POST')return route.continue();const response=await fetch(base+'/api/access',{method:'POST',headers:{'Content-Type':'application/json'},body:route.request().postData()});await route.fulfill({status:response.status,contentType:'application/json',body:await response.text()});});
  await lost.goto(base);await lost.getByRole('button',{name:'Get started',exact:true}).click();await lost.getByLabel('Date of birth',{exact:true}).fill('2000-01-01');await lost.getByRole('button',{name:'Continue',exact:true}).click();await lost.getByRole('alert').filter({hasText:'first-party cookies'}).waitFor();assert.equal(await lost.getByRole('heading',{name:'Dashboard',exact:true}).count(),0);assert.equal(await lost.getByLabel('Date of birth',{exact:true}).inputValue(),'2000-01-01');await discarded.close();console.log('PASS discarded-cookie recovery without granting unverified access');
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
