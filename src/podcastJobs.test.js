import {setAiIdentityProvider} from './aiAccess';
import {startPodcastJob,cancelPodcastJobs,cancelAllPodcastJobs,cancelOtherPodcastOwners} from './podcastJobs';
let user;
beforeEach(()=>{user={uid:'owner-one'};setAiIdentityProvider(()=>user);cancelAllPodcastJobs();});
afterEach(()=>cancelAllPodcastJobs());
const deferred=()=>{let resolve;const promise=new Promise(done=>resolve=done);return {promise,resolve};};
test('one file has one active job even if another panel tries to start it',async()=>{
  const gate=deferred(),work=jest.fn(async task=>{await gate.promise;task.check();return 'saved';});
  const first=startPodcastJob({fileId:'file',work}),second=startPodcastJob({fileId:'file',work});
  expect(first).toBe(second);gate.resolve();expect(await first).toBe('saved');expect(work).toHaveBeenCalledTimes(1);
});
test('cancelled or deleted file jobs cannot publish late audio',async()=>{
  const gate=deferred(),save=jest.fn();let signal;
  const job=startPodcastJob({fileId:'file',work:async task=>{signal=task.signal;await gate.promise;task.check();save();}});
  await Promise.resolve();cancelPodcastJobs('file');expect(signal.aborted).toBe(true);gate.resolve();await job;expect(save).not.toHaveBeenCalled();
});
test('switching accounts aborts old work and prevents a result reaching the next account',async()=>{
  const gate=deferred(),save=jest.fn();
  const job=startPodcastJob({fileId:'file',work:async task=>{await gate.promise;task.check();save();}});
  await Promise.resolve();user={uid:'owner-two'};cancelOtherPodcastOwners();gate.resolve();await job;expect(save).not.toHaveBeenCalled();
});
test('only two podcasts can run at once, protecting the shared voice service',async()=>{
  const gate=deferred(),work=()=>gate.promise;
  const a=startPodcastJob({fileId:'one',work}),b=startPodcastJob({fileId:'two',work}),extra=jest.fn();
  await startPodcastJob({fileId:'three',work:extra});expect(extra).not.toHaveBeenCalled();gate.resolve();await Promise.all([a,b]);
});
