// Server-only credential lookup. Presence and slot numbers may be reported; values never are.
const { createHash } = require('node:crypto');
const hash=value=>createHash('sha256').update(value).digest('hex');
function first(names,env){for(const name of names){const value=env[name]?.trim();if(value&&!/^(?:your[_ -]|replace[_ -]|example|placeholder)/i.test(value)&&!value.includes('...'))return value;}return '';}
function geminiCredentials(env=process.env){
  const keys=[first(['GEMINI_API_KEY','Gemini_Key'],env),first(['GEMINI_API_KEY_2','Gemini_Key_2','Gemini_Key2','GEMINI_API_KEY2'],env)],seen=new Set();
  return keys.flatMap((key,index)=>{if(!key||seen.has(key))return[];seen.add(key);const group=env['GEMINI_QUOTA_GROUP_'+(index+1)]?.trim()||'shared-project';return[{key,slot:index+1,id:hash(key),quotaGroup:hash(group)}];});
}
function providerKey(provider,env=process.env){return provider==='groq'?first(['GROQ_API_KEY','Groq_Key'],env):geminiCredentials(env)[0]?.key||'';}
function configuredNames(env=process.env){return ['GROQ_API_KEY','Groq_Key','GEMINI_API_KEY','Gemini_Key','GEMINI_API_KEY_2','Gemini_Key_2','Gemini_Key2','GEMINI_API_KEY2'].filter(name=>Boolean(env[name]?.trim()));}
function credentialId(provider,env){return provider+':'+hash(providerKey(provider,env));}
function credentialEnvironment(env,credential){return{...env,GEMINI_API_KEY:credential.key,Gemini_Key:'',GEMINI_API_KEY_2:'',Gemini_Key_2:'',Gemini_Key2:'',GEMINI_API_KEY2:'',_geminiQuotaGroup:credential.quotaGroup,_geminiSlot:credential.slot};}
module.exports={providerKey,configuredNames,credentialId,geminiCredentials,credentialEnvironment};
