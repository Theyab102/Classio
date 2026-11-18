const {test}=require('node:test'),assert=require('node:assert/strict');
const {ollamaFetch}=require('./ai-endpoints');
const {providerPolicy}=require('./ai-policy');
test('production Ollama never sends authorization to localhost, private IPs, credential URLs or an unencrypted endpoint',async()=>{
 const env={VERCEL:'1',AI_ENABLE_OLLAMA:'true',AI_OLLAMA_STUDENT_USE_APPROVED:'true',AI_OLLAMA_LICENSE_APPROVED:'true',AI_OLLAMA_FREE_TIER_CONFIRMED:'true',OLLAMA_MODEL:'qwen2.5:1.5b',OLLAMA_API_TOKEN:'fixture-private-token'};
 for(const base of ['https://127.0.0.1','https://169.254.169.254','https://10.0.0.1','https://[::1]','https://user:password@example.com'])await assert.rejects(ollamaFetch('/api/tags',{headers:{Authorization:'Bearer fixture'}},{...env,OLLAMA_BASE_URL:base}));
 for(const base of ['http://localhost:11434','http://example.com','https://user:password@example.com','https://example.com?key=ignored'])assert(!providerPolicy('ollama',{...env,OLLAMA_BASE_URL:base}).enabled);
 assert(!providerPolicy('ollama',{...env,OLLAMA_BASE_URL:'https://example.com',OLLAMA_API_TOKEN:''}).enabled);
});
