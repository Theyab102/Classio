import {useEffect,useRef} from 'react';
import {resumable} from './generationJobs';
import './aiUpgrade.css';
export default function GenerationRecovery({job,busy,onResume,onUseDraft}){
  const resume=useRef(onResume),latestJob=useRef(job);resume.current=onResume;latestJob.current=job;
  const id=job?.id,status=job?.status,retryAt=job?.retryAt||job?.failure?.retryAt||0,rateLimited=job?.failure?.status===429;
  useEffect(()=>{if(!latestJob.current||busy||!(['running','waiting','ready'].includes(status)||rateLimited))return;const timer=setTimeout(()=>resume.current(latestJob.current),Math.max(300,retryAt-Date.now()));return()=>clearTimeout(timer);},[id,status,retryAt,rateLimited,busy]);
  if(!resumable(job))return null;
  const saved=job.sections?.length||0,automatic=['running','waiting','ready'].includes(status)||rateLimited,active=busy||automatic,percent=job.total?Math.min(99,Math.floor(saved/job.total*100)):0;
  return <aside className={`generation-recovery ${active?'is-active':''}`} aria-label="Saved generation progress"><div className="generation-progress-heading"><strong>{(job.status==='waiting'||rateLimited)?'Queued to continue':active?'Generating your material':'Saved generation progress'}</strong><b>{percent}%</b></div><progress aria-label="Generation progress" max={100} value={percent}/><p role="status">{(job.status==='waiting'||rateLimited)?'Waiting for capacity. Your progress is saved and generation will continue automatically.':saved?`${saved} of ${job.total||'planned'} sections saved.`:active?'Reading the source and preparing the first section…':job.outline?'Outline saved. Continue from the next unfinished section.':'Your existing work is kept. Continue from the next unfinished section.'}</p>{!active&&<div className="studio-controls"><button className="quiet-button is-primary" onClick={()=>onResume(job)}>{!saved&&!job.outline?'Try again':'Resume generation'}</button>{saved>0&&onUseDraft&&<button className="quiet-button" onClick={()=>onUseDraft(job)}>Use saved draft</button>}</div>}</aside>;
}
