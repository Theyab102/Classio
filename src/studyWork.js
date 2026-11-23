import {useSyncExternalStore} from 'react';
import {aiOwner} from './aiAccess';

const jobs=new Map(),listeners=new Set();let host=null,snapshot=[];
const idle=Object.freeze({running:false,percent:0,progress:'',phase:'idle',error:''});
const key=(owner,file,kind)=>JSON.stringify([owner,file,kind]);
const subscribe=listener=>{listeners.add(listener);return()=>listeners.delete(listener);};
const notify=()=>{snapshot=[...jobs.values()].map(job=>job.state);listeners.forEach(listener=>listener());};
export function attachStudyWorkspace(value){host=value;return()=>{if(host===value){cancelStudyWork();host=null;}};}
export const isStudyWorkRunning=(file,kind)=>kind?!!jobs.get(key(aiOwner(),file,kind))?.state.running:[...jobs.values()].some(job=>job.state.owner===aiOwner()&&job.state.fileId===file&&job.state.running);
export const hasStudyWorkspace=()=>!!host&&host.owner===aiOwner();
export function useStudyWork(file,kind){const all=useSyncExternalStore(subscribe,()=>snapshot,()=>snapshot);return all.find(job=>job.owner===aiOwner()&&job.fileId===file&&job.kind===kind)||idle;}
export function useStudyWorks(){const all=useSyncExternalStore(subscribe,()=>snapshot,()=>snapshot);return all.filter(job=>job.owner===aiOwner()&&(job.running||job.completed||job.error));}
export function cancelStudyWork(fileId,kind){for(const job of jobs.values())if((!fileId||job.state.fileId===fileId)&&(!kind||job.state.kind===kind)){
  job.controller.abort();
  if(job.state.running&&host?.owner===job.state.owner&&host.get(job.state.fileId))host.update(job.state.fileId,current=>{const saved=current.generationJobs?.[job.state.kind];return saved?{...current,generationJobs:{...current.generationJobs,[job.state.kind]:{...saved,status:'paused',retryAt:0,failure:null,updatedAt:Date.now()+1}}}:current;});
  jobs.delete(job.key);
}notify();}
export function startStudyWork({file,kind,work,fallback}){
  const owner=aiOwner(),id=key(owner,file.id,kind),previous=jobs.get(id);
  if(previous?.state.running)return previous;
  for(const [id,item]of jobs)if(!item.state.running)jobs.delete(id);
  const workspace=hasStudyWorkspace()?host:null;
  const job={key:id,controller:new AbortController(),state:{owner,fileId:file.id,kind,name:file.name,running:true,percent:0,progress:'Reading your material…',phase:'preparing',error:''}};jobs.set(id,job);notify();
  const check=()=>{if(job.controller.signal.aborted||aiOwner()!==owner||jobs.get(id)!==job||(workspace&&(host!==workspace||!workspace.get(file.id))))throw new DOMException('Generation cancelled','AbortError');};
  const update=patch=>{check();job.state={...job.state,...patch};notify();};
  const commit=patch=>{check();if(workspace)workspace.update(file.id,patch);else fallback(patch);};
  const checkpoint=saved=>{
    commit(current=>({...current,generationJobs:{...current.generationJobs,[kind]:saved}}));
    update({phase:saved.status==='waiting'?'waiting':'writing',percent:saved.total?Math.min(99,Math.floor(saved.sections.length/saved.total*100)):0,
      progress:saved.status==='waiting'?'Waiting for capacity. Generation will continue automatically.':saved.total?`${saved.sections.length} of ${saved.total} sections saved`:'Planning your lessons…'});
  };
  job.promise=Promise.resolve().then(()=>{check();return work({signal:job.controller.signal,check,commit,checkpoint,progress:message=>update({progress:message}),finishing:()=>update({phase:'finishing',percent:99,progress:'Saving the finished material…'})});}).then(result=>{
    update({running:false,completed:true,percent:100,phase:'complete',progress:'Saved in your file'});
    setTimeout(()=>{if(jobs.get(id)===job&&!job.state.running){jobs.delete(id);notify();}},10000);return result;
  }).catch(error=>{if(jobs.get(id)===job){job.state={...job.state,running:false,phase:'failed',error:error.name==='AbortError'?'':error.message};notify();}throw error;});
  return job;
}
