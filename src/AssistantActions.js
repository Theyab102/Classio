import {useState} from 'react';
import {auth,accountHeaders} from './firebase';
import {requestJson} from './apiClient';
import {makeStudyId} from './studyTools';
import {useLocale,getReplyLanguage} from './locale';
import schema from './shared/workspace-tools';
export const actionLabels={create_notes:'Create notes',create_flashcards:'Create flashcards',create_quiz:'Create a quiz',create_course:'Create course'};
export function providerPreferences(){try{const value=JSON.parse(localStorage.getItem('classio_ai_preferences')||'{}');return {provider:['auto','groq','gemini'].includes(value.provider)?value.provider:'auto',allowFallback:value.allowFallback===true};}catch{return {provider:'auto',allowFallback:false};}}
export async function planStudyAction(text,file,signal){
  return requestJson('/api/agent',{headers:await accountHeaders(),body:{tool:'plan_action',requestId:makeStudyId(),arguments:{fileId:file.id,prompt:text,language:getReplyLanguage(),sourceText:(file.sourceText||'').slice(0,20000)},...providerPreferences()},signal});
}
export default function AssistantAction({action,onComplete,onOpenFile,getSourceText}){
  const {t}=useLocale();const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const valid=schema.workspaceToolInput({tool:action.tool,arguments:action.arguments,requestId:action.requestId,confirmed:true,...providerPreferences()});
  if(!valid)return <p role="alert">{t('This saved action is incomplete. Ask Classio to propose it again.')}</p>;
  async function execute(){
    if(busy||action.result)return;
    setBusy(true);setError('');
    try{
      const uid=auth.currentUser?.uid;
      if(!uid)throw new Error('Sign in to use account study tools.');
      const sourceText=(await getSourceText?.(valid.fileId)||'').slice(0,20000);
      const headers=await accountHeaders();
      if(auth.currentUser?.uid!==uid)throw new Error('Your account changed. Sign in again before syncing.');
      const result=await requestJson('/api/agent',{headers,body:{tool:action.tool,arguments:{...valid,sourceText},requestId:action.requestId,confirmed:true,...providerPreferences()},timeout:60000});
      if(result.saved!==true||!result.file?.id)throw new Error('Saving was not confirmed. Your original file is unchanged.');
      if(result.provider)window.dispatchEvent(new CustomEvent('classio-ai-result',{detail:{provider:result.provider,fallback:result.fallback===true}}));
      if(auth.currentUser?.uid!==uid)throw new Error('Your account changed. Sign in again before syncing.');
      onComplete({fileId:result.file.id,folderId:result.file.folderId,title:result.file.name,defaultTab:result.file.defaultTab});
    }catch(failure){setError(failure.message);}finally{setBusy(false);}
  }
  return <section className="assistant-action" aria-label={t('Review study action')}><strong>{t(actionLabels[action.tool])}</strong><p>{action.arguments.title}</p><small>{t('Creates a separate file. Your source stays intact.')}</small><details><summary>{t('Action details')}</summary><p>{action.arguments.instructions}</p><p>{t('Difficulty')}: {t(action.arguments.difficulty)}</p><p>{t('Items')}: {action.arguments.count}</p></details>{action.result?<><p role="status">{t('Saved to your account.')}</p><button type="button" disabled={busy} onClick={async()=>{setBusy(true);setError('');try{await onOpenFile?.(action.result);}catch(failure){setError(failure.message);}finally{setBusy(false);}}}>{t('Open saved file')}</button></>:<button type="button" disabled={busy} onClick={execute}>{t(busy?'Creating and saving…':error?'Retry action':'Confirm and create')}</button>}{error&&<p role="alert">{t(error)}</p>}</section>;
}
