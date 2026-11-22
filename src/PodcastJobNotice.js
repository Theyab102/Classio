import { cancelPodcastJobs, usePodcastJobs } from './podcastJobs';
export default function PodcastJobNotice() {
  const jobs = usePodcastJobs();
  if (!jobs.length) return null;
  return <aside className="podcast-background-notice" aria-label="Background podcast generation">
    {jobs.map(job => <div key={job.fileId}><div role="status"><strong>Preparing podcast · {job.name}</strong><span>{job.progress || 'Starting…'}</span></div><button type="button" className="text-button" onClick={() => cancelPodcastJobs(job.fileId)}>Cancel</button></div>)}
  </aside>;
}
