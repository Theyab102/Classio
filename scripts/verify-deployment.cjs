// Read-only checks by default; --verification-only also exercises a fresh,
// fictional adult guest cookie flow. No accounts, uploads or AI jobs are created.
const assert=require('node:assert/strict');
const origin=new URL(process.argv[2]||'http://127.0.0.1:3000').origin;
const verificationOnly=process.argv.includes('--verification-only');
let cookie='';
async function access(body){
 const response=await fetch(origin+'/api/access',{method:body?'POST':'GET',headers:{Accept:'application/json',...(body?{'Content-Type':'application/json'}:{}),...(cookie?{Cookie:cookie}:{})},...(body?{body:JSON.stringify(body)}:{}),signal:AbortSignal.timeout(15000),redirect:'error',cache:'no-store'});
 const result=await response.json().catch(()=>{throw new Error('The access route returned HTML or invalid JSON. Deploy the Vercel API routes, not just static files.');});
 if(!response.ok)throw new Error('Access failed: HTTP '+response.status+' '+(result.code||'')+'. Check the server deployment and CLASSIO_SESSION_SECRET setting.');
 const signed=response.headers.get('set-cookie');if(signed)cookie=signed.split(';')[0];return result;
}
(async()=>{
 const first=await access();assert.equal(first.stage,'new');console.log('PASS access endpoint returns a valid new-session decision');
 if(verificationOnly){
  assert.equal((await access({action:'screen',birthDate:'2000-01-01'})).stage,'eligible');
  assert.equal((await access()).stage,'eligible');
  assert.equal((await access({action:'accept',termsAccepted:true})).stage,'accepted');
  assert.equal((await access()).stage,'accepted');
  console.log('PASS server screening, terms, signed-cookie retention and returning guest access');
 }else{
  const response=await fetch(origin+'/api/study',{signal:AbortSignal.timeout(15000)}),result=await response.json();
  assert(response.ok&&result.providers&&(result.providers.groq||result.providers.gemini),'No enabled AI provider is reported. Configure provider/Admin/quota settings and redeploy.');
  console.log('PASS AI configuration metadata reports an enabled provider (no live generation test performed)');
 }
})().catch(error=>{console.error(error.message);process.exitCode=1;});
