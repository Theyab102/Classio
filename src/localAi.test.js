import {createLocalWorker} from './localAiWorkerFactory';
import {clearLocalModels,disableLocalAi,loadLocalModel,localAiState,semanticSearch} from './localAi';
import {setAiIdentityProvider} from './aiAccess';
jest.mock('./localAiWorkerFactory',()=>({createLocalWorker:jest.fn()}));
let worker,reply,owner;
beforeEach(()=>{disableLocalAi();owner='fixture-alice';setAiIdentityProvider(()=>({uid:owner}));reply=data=>({id:data.id,result:true});createLocalWorker.mockReset();createLocalWorker.mockImplementation(()=>{worker={terminate:jest.fn(),postMessage:jest.fn(data=>Promise.resolve().then(()=>worker.onmessage?.({data:reply(data)})))};return worker;});});
afterEach(disableLocalAi);
test('removing cached models starts a cleanup worker even after refresh, without requesting inference/model loading',async()=>{
 await clearLocalModels();expect(createLocalWorker).toHaveBeenCalledTimes(1);expect(worker.postMessage.mock.calls.map(([data])=>data.type)).toEqual(['clear-cache']);expect(worker.terminate).toHaveBeenCalledTimes(1);expect(localAiState()).toMatchObject({llm:false,embeddings:false,loading:false});
});
test('a failed embedding reload cannot leave the settings claiming that semantic search is ready',async()=>{
 await loadLocalModel('embeddings');expect(localAiState().embeddings).toBe(true);reply=data=>({id:data.id,error:'Model unavailable'});await expect(loadLocalModel('embeddings')).rejects.toThrow('Model unavailable');expect(localAiState()).toMatchObject({embeddings:false,loading:false});expect(await semanticSearch('query',[])).toBeNull();
});
test('a late local result is rejected when the signed-in account changes',async()=>{
 let send;reply=data=>{send=data;return{ id:data.id,progress:20 };};const pending=loadLocalModel('embeddings');while(!send)await Promise.resolve();owner='fixture-bob';worker.onmessage({data:{id:send.id,result:true}});await expect(pending).rejects.toMatchObject({name:'AbortError'});expect(localAiState().embeddings).toBe(false);
});
