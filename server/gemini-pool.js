const{credentialId}=require('./ai-settings');
const limits=new Map();
const usage=new Map();
// Reserve estimated work before awaiting a provider. The task preference breaks
// ties; recent work and concurrent reservations keep one free key from taking
// every fallback. Cooldowns still apply to the actual shared project quota.
function planGemini(credentials,input){
 const now=Date.now(),preferred=input.audio||input.vision||['conversation','style','classification'].includes(input.task)?2:1;
 return [...credentials].sort((a,b)=>{
  const score=key=>{const entry=usage.get(key.id);return(entry&&entry.until>now?entry.tokens:0)+(key.slot===preferred?0:.5);};
  return score(a)-score(b);
 });
}
function reserveGemini(credential,input){
 const now=Date.now(),previous=usage.get(credential.id),text=(input.messages||[]).reduce((sum,message)=>sum+(typeof message.content==='string'?message.content.length:message.content.filter(part=>part.type==='text').reduce((n,part)=>n+part.text.length,0)),0);
 const tokens=(input.audio?4000:input.vision?1500:0)+Math.ceil(text/4)+(input.maxTokens||1000);
 usage.set(credential.id,{tokens:(previous?.until>now?previous.tokens:0)+tokens,until:previous?.until>now?previous.until:now+60000});
 if(usage.size>64)for(const[key,value]of usage)if(value.until<=now)usage.delete(key);
}
function geminiBlocked(env){const now=Date.now(),quota=limits.get('quota:'+env._geminiQuotaGroup),key=limits.get(credentialId('gemini',env));return quota?.until>now?quota:key?.until>now?key:null;}
function noteGeminiFailure(env,status,seconds=60){if(![401,403,429].includes(status))return;const retryAfter=Math.min(600,Math.max(1,seconds||60));if(limits.size>64){for(const[key,value]of limits)if(value.until<=Date.now())limits.delete(key);}limits.set(status===429?'quota:'+env._geminiQuotaGroup:credentialId('gemini',env),{until:Date.now()+retryAfter*1000,code:status===429?'PROVIDER_RATE_LIMIT':'CREDENTIALS_REJECTED'});}
module.exports={geminiBlocked,noteGeminiFailure,planGemini,reserveGemini,clearGeminiPool:()=>{limits.clear();usage.clear();}};
