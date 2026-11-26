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

test('a queued request cannot be sent using a different account after acquiring its slot',async()=>{
 const previous=global.fetch,firstController=new AbortController(),secondController=new AbortController();let release;
 global.fetch=jest.fn((_url,options)=>new Promise((resolve,reject)=>{release=()=>resolve({ok:true,status:200,json:async()=>({text:'Old account response'})});options.signal.addEventListener('abort',()=>reject(new DOMException('Cancelled','AbortError')),{once:true});}));
 const first=sendStudyRequest({messages:[{role:'user',content:'First private request'}],maxTokens:100},firstController.signal).catch(error=>error);await flush();
 const second=sendStudyRequest({messages:[{role:'user',content:'Queued private material'}],maxTokens:100},secondController.signal).catch(error=>error);
 setAiIdentityProvider(()=>({uid:'different-account',getIdToken:async()=> 'fixture-other-token'}));release();
 try{await flush();expect(global.fetch).toHaveBeenCalledTimes(1);expect(await first).toHaveProperty('name','AbortError');expect(await second).toHaveProperty('name','AbortError');}finally{firstController.abort();secondController.abort();await Promise.all([first,second]);global.fetch=previous;}
});
test('a transcription capacity wait leaves text generation available',async()=>{
 jest.useFakeTimers();const previous=global.fetch,controller=new AbortController();global.fetch=jest.fn(async(_url,options)=>JSON.parse(options.body).kind==='audio'?{ok:false,status:429,headers:{get:()=> '45'},json:async()=>({code:'PROVIDER_RATE_LIMIT'})}:{ok:true,status:200,json:async()=>({text:'Text is available'})});
 const audio=sendStudyRequest({kind:'audio',audio:'dGVzdA==',filename:'recording.webm'},controller.signal),cancelled=expect(audio).rejects.toHaveProperty('name','AbortError');
 try{await flush();const text=sendStudyRequest({messages:[{role:'user',content:'Explain gravity'}],maxTokens:100});await flush();expect(global.fetch).toHaveBeenCalledTimes(2);controller.abort();await cancelled;await expect(text).resolves.toBe('Text is available');}finally{controller.abort();global.fetch=previous;jest.useRealTimers();}
});

test('a vision-provider limit leaves text available while a shared app quota waits across both',async()=>{
 jest.useFakeTimers();const previous=global.fetch;
 try{for(const code of ['PROVIDER_RATE_LIMIT','CLIENT_RATE_LIMIT']){
  clearStudyQueue();const visionController=new AbortController(),textController=new AbortController();let requests=0;global.fetch=jest.fn(async()=>++requests===1?{ok:false,status:429,headers:{get:()=> '45'},json:async()=>({code})}:{ok:true,status:200,json:async()=>({text:'Ready'})});
  const vision=sendStudyRequest({messages:[{role:'user',content:[{type:'image_url',image_url:{url:'data:image/png;base64,dGVzdA=='}},{type:'text',text:'Explain the diagram'}]}],maxTokens:100},visionController.signal).catch(error=>error);await flush();
  const text=sendStudyRequest({messages:[{role:'user',content:'Explain gravity'}],maxTokens:100},textController.signal).catch(error=>error);await flush();expect(requests).toBe(code==='PROVIDER_RATE_LIMIT'?2:1);
  if(code==='PROVIDER_RATE_LIMIT')expect(await text).toBe('Ready');
  visionController.abort();textController.abort();await Promise.all([vision,text]);
 }}finally{global.fetch=previous;jest.useRealTimers();}
});
