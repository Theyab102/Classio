import {aiOwner} from './aiAccess';

const observers=new WeakMap(),lanes=new Map(),deadlines=new Map();
const aborted=()=>new DOMException('Generation cancelled','AbortError');
export function bindStudyWait(signal,observer){if(!signal)return()=>{};observers.set(signal,observer);return()=>observers.delete(signal);}
export async function reportStudyWait(signal,event){await observers.get(signal)?.(event);}
export function retryDelay(value,now=Date.now()){
  const seconds=Number(value)||Math.ceil((Date.parse(value)-now)/1000);
  return Number.isFinite(seconds)&&seconds>0?Math.min(7*86400,seconds):30;
}
export async function waitForStudyCapacity(until,signal,owner=aiOwner()){
  const check=()=>{if(signal?.aborted||aiOwner()!==owner)throw aborted();};
  check();await reportStudyWait(signal,{until,phase:'waiting'});
  try{
    while(Date.now()<until){
      check();await new Promise((resolve,reject)=>{
        const cancel=()=>{clearTimeout(timer);signal?.removeEventListener('abort',cancel);reject(aborted());};
        const timer=setTimeout(()=>{signal?.removeEventListener('abort',cancel);resolve();},Math.min(20000,until-Date.now()));
        signal?.addEventListener('abort',cancel,{once:true});
      });
      check();await reportStudyWait(signal,{until,phase:'heartbeat'});
    }
  }finally{if(!signal?.aborted&&aiOwner()===owner)await reportStudyWait(signal,{until:0,phase:'running'});}
  check();
}
const scopeKey=(owner,scope='study')=>JSON.stringify([owner,scope]);
export const studyCapacityDeadline=(owner,scope)=>deadlines.get(scopeKey(owner,scope))||0;
export function deferStudyCapacity(owner,seconds,scope){const key=scopeKey(owner,scope),until=Date.now()+seconds*1000;deadlines.set(key,Math.max(until,deadlines.get(key)||0));return deadlines.get(key);}
export function acquireStudySlot(owner,signal,scope){
  if(signal?.aborted)return Promise.reject(aborted());
  const key=scopeKey(owner,scope),lane=lanes.get(key)||{busy:false,queue:[]};lanes.set(key,lane);
  return new Promise((resolve,reject)=>{
    const entry={grant:()=>{
      signal?.removeEventListener('abort',cancel);lane.busy=true;let released=false;
      resolve(()=>{if(released)return;released=true;const next=lane.queue.shift();if(next)next.grant();else{lane.busy=false;lanes.delete(key);}});
    }};
    const cancel=()=>{lane.queue=lane.queue.filter(item=>item!==entry);reject(aborted());};
    if(lane.busy){lane.queue.push(entry);signal?.addEventListener('abort',cancel,{once:true});}else entry.grant();
  });
}
export function clearStudyQueue(){deadlines.clear();}
