import {setAiIdentityProvider} from './aiAccess';
import {sendStudyRequest} from './studyTools';
import {clearStudyQueue,bindStudyWait} from './studyQueue';
const flush=async()=>{for(let i=0;i<24;i++)await Promise.resolve();};
beforeEach(()=>{setAiIdentityProvider(()=>({uid:'fixture-'+expect.getState().currentTestName,getIdToken:()=>Promise.resolve('fixture-account-token')}));clearStudyQueue();});
test('a stalled study request times out and releases its queue slot',async()=>{
 jest.useFakeTimers();const previous=global.fetch;global.fetch=jest.fn((_url,options)=>new Promise((_resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')))));
 try{const pending=sendStudyRequest({messages:[{role:'user',content:'Explain gravity'}],maxTokens:100});const result=expect(pending).rejects.toThrow('took too long');await flush();expect(global.fetch).toHaveBeenCalledTimes(1);jest.advanceTimersByTime(75000);await result;expect(jest.getTimerCount()).toBe(0);}finally{global.fetch=previous;jest.useRealTimers();}
});
test('an interrupted queued request preserves cancellation and never reaches a provider',async()=>{
 const previous=global.fetch,controller=new AbortController();global.fetch=jest.fn();
 try{const pending=sendStudyRequest({},controller.signal);controller.abort();await expect(pending).rejects.toHaveProperty('name','AbortError');expect(global.fetch).not.toHaveBeenCalled();}finally{global.fetch=previous;}
});
test('rate limits wait through the advertised reset and resume the same request automatically',async()=>{
 jest.useFakeTimers();const previous=global.fetch,controller=new AbortController(),states=[];const unbind=bindStudyWait(controller.signal,event=>states.push(event));
 global.fetch=jest.fn().mockResolvedValueOnce({ok:false,status:429,headers:{get:()=> '12'},json:async()=>({code:'PROVIDER_RATE_LIMIT',error:'Rate limited'})}).mockResolvedValueOnce({ok:true,status:200,json:async()=>({text:'Finished automatically'})});
 try{const pending=sendStudyRequest({messages:[{role:'user',content:'Explain gravity'}],maxTokens:100},controller.signal);await flush();expect(global.fetch).toHaveBeenCalledTimes(1);expect(states.some(state=>state.phase==='waiting')).toBe(true);jest.advanceTimersByTime(11999);await flush();expect(global.fetch).toHaveBeenCalledTimes(1);jest.advanceTimersByTime(1);await flush();await expect(pending).resolves.toBe('Finished automatically');expect(global.fetch).toHaveBeenCalledTimes(2);expect(jest.getTimerCount()).toBe(0);}finally{controller.abort();unbind();global.fetch=previous;jest.useRealTimers();}
});
test('cancelling a capacity wait prevents any later retry',async()=>{
 jest.useFakeTimers();const previous=global.fetch,controller=new AbortController();global.fetch=jest.fn().mockResolvedValue({ok:false,status:429,headers:{get:()=> '86400'},json:async()=>({code:'RATE_LIMITED'})});
 try{const pending=sendStudyRequest({},controller.signal);const rejected=expect(pending).rejects.toHaveProperty('name','AbortError');await flush();controller.abort();await rejected;jest.advanceTimersByTime(86400000);await flush();expect(global.fetch).toHaveBeenCalledTimes(1);expect(jest.getTimerCount()).toBe(0);}finally{global.fetch=previous;jest.useRealTimers();}
});
