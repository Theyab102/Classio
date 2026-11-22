import { useSyncExternalStore } from 'react';
import { aiOwner } from './aiAccess';

const jobs = new Map(), listeners = new Set();
const idle = Object.freeze({ phase: 'idle', progress: '', error: '', running: false });
let snapshot = [];
const keyFor = (owner, file) => JSON.stringify([owner, file]);
const subscribe = listener => { listeners.add(listener); return () => listeners.delete(listener); };
function notify() {
  snapshot = [...jobs.values()].map(job => job.state);
  listeners.forEach(listener => listener());
}
export function usePodcastJob(fileId) {
  const key = keyFor(aiOwner(), fileId);
  return useSyncExternalStore(subscribe, () => jobs.get(key)?.state || idle, () => idle);
}
export function usePodcastJobs() {
  const all = useSyncExternalStore(subscribe, () => snapshot, () => snapshot);
  return all.filter(job => job.owner === aiOwner() && job.running);
}
export function cancelPodcastJobs(fileId, owner = aiOwner()) {
  for (const job of jobs.values()) {
    if (job.state.owner === owner && (!fileId || job.state.fileId === fileId)) {
      job.controller.abort();
      job.state = { ...job.state, phase: 'idle', running: false, progress: '', error: '' };
    }
  }
  notify();
}
export function cancelOtherPodcastOwners(owner = aiOwner()) {
  for (const job of jobs.values()) {
    if (job.state.owner !== owner) { job.controller.abort(); jobs.delete(job.key); }
  }
  notify();
}
export function cancelAllPodcastJobs() {
  for (const job of jobs.values()) job.controller.abort();
  jobs.clear();
  notify();
}

// Work belongs to the account/file, not the mounted podcast panel. Functional
// workspace updates merge into the current file even after navigation or edits.
export function startPodcastJob({ fileId, name, phase, work }) {
  const owner = aiOwner(), key = keyFor(owner, fileId), previous = jobs.get(key);
  if (previous?.state.running) return previous.promise;
  // Retain only small status snapshots for completed jobs, never their sources.
  if (jobs.size >= 30) {
    for (const [id, job] of jobs) if (!job.state.running) jobs.delete(id);
  }
  const job = { key, controller: new AbortController(), state: { owner, fileId, name, phase, progress: '', error: '', running: true } };
  jobs.set(key, job);
  if (snapshot.filter(item => item.running).length >= 2) {
    job.state = { ...job.state, phase: 'idle', running: false, error: 'Two podcasts are already being prepared. Wait for one to finish or cancel it.' };
    notify();
    return Promise.resolve(null);
  }
  const check = () => {
    if (job.controller.signal.aborted || aiOwner() !== owner || jobs.get(key) !== job)
      throw new DOMException('Podcast generation cancelled', 'AbortError');
  };
  const update = patch => { check(); job.state = { ...job.state, ...patch }; notify(); };
  notify();
  job.promise = Promise.resolve().then(async () => {
    check();
    return work({ signal: job.controller.signal, check, update });
  }).then(result => {
    check();
    update({ phase: 'idle', running: false, progress: '', completed: true });
    return result;
  }).catch(error => {
    if (jobs.get(key) === job && aiOwner() === owner) {
      job.state = { ...job.state, phase: 'idle', running: false, progress: '',
        error: error.name === 'AbortError' ? '' : error.message || 'Podcast generation could not finish. Your saved episodes are kept.' };
      notify();
    }
    return null;
  });
  return job.promise;
}
