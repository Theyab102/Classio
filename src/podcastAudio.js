import { requestJson } from './apiClient';
import { accountHeaders } from './firebase';
import { PODCAST_SERVER_URL } from './studyTools';
import schema from './shared/podcast-schema';

export function parseDialogue(raw, hosts=2){
  let value;
  try{value=JSON.parse(raw.replace(/^\s*```(?:json)?\s*|\s*```\s*$/g,''));}catch{throw new Error('The AI returned an unreadable podcast script. Your previous script is kept; retry.');}
  if(!Array.isArray(value))value=value?.dialogue;
  const input=schema.podcastInput({language:'en',voices:hosts===1?['aria']:['aria','echo'],speed:1,dialogue:value});
  if(!input)throw new Error('The AI returned an incomplete dialogue. Your previous script is kept; retry.');
  return input.dialogue;
}
export function singleVoiceDialogue(script){
  const phrases=script.match(/[^.!?]+[.!?]*|[.!?]+/g)||[script];const dialogue=[];
  for(const phrase of phrases){const text=phrase.trim();if(!text)continue;if(text.length>1200)throw new Error('Split this long paragraph into shorter sentences before generating audio.');const last=dialogue.at(-1);if(last&&last.text.length+text.length<1150)last.text+=' '+text;else dialogue.push({speaker:0,text});}
  return dialogue;
}
function delay(ms,signal){return new Promise((resolve,reject)=>{const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve();},ms);if(signal?.aborted)abort();else signal?.addEventListener('abort',abort,{once:true});});}
export async function generatePodcastAudio(payload,{signal,onProgress}={}){
  const input=schema.podcastInput(payload);if(!input)throw new Error('Check the English dialogue and distinct host voices before generating audio.');
  if(!PODCAST_SERVER_URL)throw new Error('The voice server URL is not configured. Your script is kept.');
  let authorization;
  const renew=async()=>{authorization=await requestJson('/api/podcast',{body:input,headers:await accountHeaders(),signal,timeout:20000});};
  await renew();
  const headers=()=>({Authorization:'Bearer '+authorization.ticket});
  let job=await requestJson(PODCAST_SERVER_URL+'/podcasts/jobs',{body:input,headers:headers(),signal,timeout:20000});
  const deadline=Date.now()+20*60*1000;
  while(job.status!=='complete'){
    if(job.status==='failed')throw new Error(job.error||'Audio generation failed. Your script is kept; retry.');
    onProgress?.(job);if(Date.now()>deadline)throw new Error('The job is still on the voice server. Your script is kept; reopen Generate audio to resume checking it.');
    await delay(1800,signal);if(Date.now()>authorization.expiresAt-15000)await renew();
    job=await requestJson(PODCAST_SERVER_URL+'/podcasts/jobs/'+job.id,{headers:headers(),signal,timeout:20000});
  }
  onProgress?.(job);
  if(Date.now()>authorization.expiresAt-15000)await renew();
  const controller=new AbortController(),abort=()=>controller.abort();signal?.addEventListener('abort',abort,{once:true});if(signal?.aborted)abort();
  const timer=setTimeout(abort,90000);
  try{
    const response=await fetch(PODCAST_SERVER_URL+'/podcasts/jobs/'+job.id+'/audio',{headers:headers(),signal:controller.signal,cache:'no-store',credentials:'omit',referrerPolicy:'no-referrer'});
    if(!response.ok||!response.headers.get('content-type')?.startsWith('audio/'))throw new Error('The completed audio could not be downloaded. Retry to reuse the saved job.');
    return {blob:await response.blob(),metadata:job.metadata,id:job.id};
  }finally{clearTimeout(timer);signal?.removeEventListener('abort',abort);}
}
function audioDatabase(){return new Promise((resolve,reject)=>{const request=indexedDB.open('classio-podcast-audio',1);request.onupgradeneeded=()=>request.result.createObjectStore('audio');request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);});}
export async function cachePodcast(key,blob){const db=await audioDatabase();try{await new Promise((resolve,reject)=>{const transaction=db.transaction('audio','readwrite');transaction.objectStore('audio').put(blob,key);transaction.oncomplete=resolve;transaction.onerror=()=>reject(transaction.error);transaction.onabort=()=>reject(transaction.error);});}finally{db.close();}}
export async function readPodcast(key){const db=await audioDatabase();try{return await new Promise((resolve,reject)=>{const request=db.transaction('audio').objectStore('audio').get(key);request.onsuccess=()=>resolve(request.result||null);request.onerror=()=>reject(request.error);});}finally{db.close();}}
export function captionAt(captions,time){return captions.findIndex(cue=>time>=cue.start&&time<cue.end);}
let guestDraftSession;
export function voiceDraftKey(scope,uid){
  if(!uid){
    try{guestDraftSession=sessionStorage.getItem('classio_voice_draft_session')||guestDraftSession||crypto.randomUUID();sessionStorage.setItem('classio_voice_draft_session',guestDraftSession);}catch{guestDraftSession ||= String(Date.now())+Math.random();}
  }
  return 'voice-draft:'+(uid||'guest-'+guestDraftSession)+':'+scope;
}
