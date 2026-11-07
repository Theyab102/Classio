const{credentialId}=require('./ai-settings');
const limits=new Map();
function geminiBlocked(env){const now=Date.now(),quota=limits.get('quota:'+env._geminiQuotaGroup),key=limits.get(credentialId('gemini',env));return quota?.until>now?quota:key?.until>now?key:null;}
function noteGeminiFailure(env,status,seconds=60){if(![401,403,429].includes(status))return;const retryAfter=Math.min(600,Math.max(1,seconds||60));if(limits.size>64){for(const[key,value]of limits)if(value.until<=Date.now())limits.delete(key);}limits.set(status===429?'quota:'+env._geminiQuotaGroup:credentialId('gemini',env),{until:Date.now()+retryAfter*1000,code:status===429?'PROVIDER_RATE_LIMIT':'CREDENTIALS_REJECTED'});}
module.exports={geminiBlocked,noteGeminiFailure,clearGeminiPool:()=>limits.clear()};
