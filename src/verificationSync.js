import {auth,linkVerification} from './firebase';

// Account linking never delays workspace loading. Retried work remains scoped
// to the signed-in UID and is cancelled on sign-out or account changes.
export function watchVerificationSync(uid,onStatus){
  let closed=false,attempts=0,timer,controller;
  async function run(){
    if(closed||controller||auth.currentUser?.uid!==uid)return;
    controller=new AbortController();attempts++;
    try{await linkVerification({signal:controller.signal});if(!closed&&auth.currentUser?.uid===uid)onStatus('synced');}
    catch{if(!closed&&auth.currentUser?.uid===uid){if(attempts<3)timer=setTimeout(run,attempts===1?5000:20000);else onStatus('pending');}}
    finally{controller=null;}
  }
  const online=()=>{clearTimeout(timer);attempts=0;run();};
  window.addEventListener('online',online);run();
  return()=>{closed=true;clearTimeout(timer);controller?.abort();window.removeEventListener('online',online);};
}
