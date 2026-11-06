import { requestJson } from './apiClient';

const stages=new Set(['new','eligible','accepted','blocked']);
function stageOf(value){
  if(!stages.has(value?.stage))throw Object.assign(new Error('Classio could not confirm access. Please try again.'),{code:'INVALID_ACCESS_RESPONSE'});
  return value.stage;
}
export async function accessStage(body,{headers={},signal}={}){
  const stage=stageOf(await requestJson('/api/access',{body,headers:{Accept:'application/json',...headers},signal,timeout:7000}));
  // A POST response alone does not establish that Safari retained the cookie.
  if(body&&stage!=='new'){
    const confirmed=stageOf(await requestJson('/api/access',{signal,timeout:7000}));
    if(confirmed!==stage)throw Object.assign(new Error('Allow first-party cookies for Classio, then try again. Your browser did not keep the access decision.'),{code:'ACCESS_COOKIE_REQUIRED'});
  }
  return stage;
}
export function untilAborted(promise,signal){
  return new Promise((resolve,reject)=>{
    const abort=()=>reject(new DOMException('Aborted','AbortError'));
    if(signal.aborted){abort();return;}
    signal.addEventListener('abort',abort,{once:true});
    Promise.resolve(promise).then(resolve,reject).finally(()=>signal.removeEventListener('abort',abort));
  });
}
export async function restoreAccess(signal){
  let stage=await accessStage(undefined,{signal});
  let requested=false;try{requested=localStorage.getItem('classio_google_signin_requested')==='true';}catch{}
  if(stage==='new'&&requested){
    const firebase=await untilAborted(import('./firebase'),signal);
    const user=await untilAborted(firebase.restoreSignedInAccount(),signal);
    if(user)stage=await accessStage({action:'restore'},{headers:await untilAborted(firebase.accountHeaders(),signal),signal});
  }
  return stage;
}
export function accessFailure(error){
  if(error.code==='VERIFICATION_CONFIG_MISSING')return 'Classio is temporarily unavailable. Please check back later.';
  if(error.code==='ACCESS_COOKIE_REQUIRED')return error.message;
  if(error.code==='INVALID_ACCESS_RESPONSE')return 'Classio’s access service could not be reached. Please try again shortly.';
  if(['INVALID_DATE','TERMS_REQUIRED','AGE_RESTRICTED','VERIFICATION_UNAVAILABLE'].includes(error.code))return error.message;
  return 'We couldn’t connect to Classio. Your saved verification has not been changed. We’ll try again when you’re online.';
}
