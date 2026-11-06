import { useCallback, useEffect, useRef, useState } from 'react';
import { accessFailure, accessStage, restoreAccess, untilAborted } from './accessSession';

export default function useAccessSession(enabled){
  const [stage,setStage]=useState('checking'),[checking,setChecking]=useState(false),[pending,setPending]=useState(false),[error,setError]=useState('');
  const [errorCode,setErrorCode]=useState('');
  const operation=useRef(null),alive=useRef(false),lastCheck=useRef(0),currentStage=useRef(stage),failureCode=useRef('');
  currentStage.current=stage;
  const check=useCallback(async()=>{
    if(!enabled||operation.current)return;
    const controller=new AbortController();operation.current=controller;lastCheck.current=Date.now();setChecking(true);
    const timer=setTimeout(()=>controller.abort(),11000);
    try{
      const result=await restoreAccess(controller.signal);
      if(alive.current&&!controller.signal.aborted){setStage(result);setError('');setErrorCode('');failureCode.current='';}
    }catch(failure){
      if(alive.current&&operation.current===controller){failureCode.current=failure.code||'';setErrorCode(failureCode.current);setError(accessFailure(failure));if(currentStage.current==='checking')setStage('unavailable');}
    }finally{
      clearTimeout(timer);
      if(operation.current===controller){operation.current=null;if(alive.current)setChecking(false);}
    }
  },[enabled]);
  useEffect(()=>{
    if(!enabled)return;
    alive.current=true;
    check();
    // A single delayed retry handles a transient cold start without endless polling.
    const retry=setTimeout(()=>{if(currentStage.current==='unavailable'&&failureCode.current!=='VERIFICATION_CONFIG_MISSING')check();},14000);
    const online=()=>{if(currentStage.current==='unavailable')check();};
    const visible=()=>{if(document.visibilityState==='visible'&&currentStage.current==='unavailable'&&Date.now()-lastCheck.current>5000)check();};
    window.addEventListener('online',online);document.addEventListener('visibilitychange',visible);
    return()=>{alive.current=false;operation.current?.abort();operation.current=null;clearTimeout(retry);window.removeEventListener('online',online);document.removeEventListener('visibilitychange',visible);};
  },[enabled,check]);
  const run=async(body,authenticate)=>{
    if(operation.current)return false;
    const controller=new AbortController();operation.current=controller;setPending(true);setError('');
    // Popup sign-in is user-controlled; bound the server/auth restoration afterwards.
    let timer;
    try{
      timer=setTimeout(()=>controller.abort(),60000);
      const headers=authenticate?await untilAborted(authenticate(controller.signal),controller.signal):{};
      clearTimeout(timer);
      timer=setTimeout(()=>controller.abort(),16000);
      const result=await accessStage(body,{headers:await untilAborted(Promise.resolve(headers),controller.signal),signal:controller.signal});
      if(alive.current&&!controller.signal.aborted){setStage(result);setErrorCode('');failureCode.current='';return true;}
    }catch(failure){if(alive.current){failureCode.current=failure.code||'';setErrorCode(failureCode.current);setError(failure.code?.startsWith('auth/')?failure.message:accessFailure(failure));}}
    finally{clearTimeout(timer);if(operation.current===controller){operation.current=null;if(alive.current)setPending(false);}}
    return false;
  };
  return {stage,checking,pending,error,configurationMissing:errorCode==='VERIFICATION_CONFIG_MISSING',check,run};
}
