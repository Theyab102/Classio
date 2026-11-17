import {useEffect,useSyncExternalStore} from 'react';
let context;const listeners=new Set();
const enabled=()=>{try{return localStorage.getItem('classio_study_sounds')!=='off';}catch{return true;}};
export function unlockStudyAudio(){try{const Context=window.AudioContext||window.webkitAudioContext;if(Context){context??=new Context();context.resume().catch(()=>{});}}catch{}}
export function playStudySound(kind){
  if(!enabled())return;try{unlockStudyAudio();if(!context||context.state!=='running')return;const notes=kind==='correct'?[523.25,783.99]:kind==='complete'?[523.25,659.25,783.99]:[220,174.61];
    notes.forEach((frequency,index)=>{const oscillator=context.createOscillator(),gain=context.createGain(),at=context.currentTime+index*.085;oscillator.type='sine';oscillator.frequency.value=frequency;gain.gain.setValueAtTime(.0001,at);gain.gain.exponentialRampToValueAtTime(.07,at+.012);gain.gain.exponentialRampToValueAtTime(.0001,at+.18);oscillator.connect(gain);gain.connect(context.destination);oscillator.start(at);oscillator.stop(at+.2);oscillator.onended=()=>{oscillator.disconnect();gain.disconnect();};});
  }catch{}
}
function subscribe(listener){listeners.add(listener);window.addEventListener('storage',listener);return()=>{listeners.delete(listener);window.removeEventListener('storage',listener);};}
export function StudySoundToggle(){const on=useSyncExternalStore(subscribe,enabled,()=>true);useEffect(()=>{document.addEventListener('pointerdown',unlockStudyAudio,{passive:true});document.addEventListener('keydown',unlockStudyAudio);return()=>{document.removeEventListener('pointerdown',unlockStudyAudio);document.removeEventListener('keydown',unlockStudyAudio);};},[]);return <button type="button" className="quiet-button study-sound-toggle" aria-label="Study sound effects" aria-pressed={on} onClick={()=>{try{localStorage.setItem('classio_study_sounds',on?'off':'on');}catch{}for(const notify of listeners)notify();if(!on)playStudySound('correct');}}>Sound {on?'on':'off'}</button>;}
