const { test, beforeEach } = require("node:test");
const assert = require("node:assert/strict");
const { availableProviders, geminiPayload, generateStudy, clearProviderCaches } = require("./ai-providers");

beforeEach(() => clearProviderCaches());

const env = { GROQ_TEXT_MODEL: "test-groq-model", GROQ_API_KEY: "test-groq", GEMINI_API_KEY: "test-google", GEMINI_ENABLED: "true", GEMINI_MODEL: "test-model" };
const input = { messages: [{ role: "system", content: "Use the learner's language." }, { role: "user", content: "Explain gravity." }], maxTokens: 500 };

test("Gemini requires explicit enablement, a key; discovers a model when no override is set", () => {
  assert.deepEqual(availableProviders({ ...env, GEMINI_ENABLED: "false" }), ["groq"]);
  assert.deepEqual(availableProviders({ ...env, GEMINI_MODEL: "" }), ["groq", "gemini"]);
  assert.deepEqual(availableProviders({ ...env, GEMINI_MODEL: "../other/key" }), ["groq"]);
  assert.deepEqual(availableProviders({ ...env, AI_PROVIDER_ORDER: "gemini,gemini,unknown,groq" }), ["gemini", "groq"]);
});

test("Gemini conversion preserves system instructions, conversation roles and image bytes", () => {
  const payload = geminiPayload({ ...input, messages: [...input.messages,
    { role: "assistant", content: "Gravity attracts masses." },
    { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,aGVsbG8=" } }, { type: "text", text: "Explain this diagram." }] },
  ] });
  assert.equal(payload.systemInstruction.parts[0].text, input.messages[0].content);
  assert.deepEqual(payload.contents.map(item => item.role), ["user", "model", "user"]);
  assert.deepEqual(payload.contents[2].parts[0], { inlineData: { mimeType: "image/png", data: "aGVsbG8=" } });
  assert.equal(payload.generationConfig.maxOutputTokens, 500);
});

test("Groq rate limiting falls back to Gemini without sending keys in URLs or returning thought text", async () => {
  const calls = [], reports = [];
  const result = await generateStudy(input, { env, report: item => reports.push(item), fetcher: async (url, options) => {
    calls.push({ url, options });
    if (calls.length === 1) return { ok: false, status: 429, json: async () => ({}) };
    return { ok: true, json: async () => ({ candidates: [{ finishReason: "STOP", content: { parts: [
      { thought: true, text: "Private reasoning" }, { text: "Gravity attracts masses." },
    ] } }] }) };
  } });
  assert.deepEqual(result, { provider: "gemini", text: "Gravity attracts masses." });
  assert.equal(calls.length, 2);
  assert.equal(calls[1].options.headers["x-goog-api-key"], env.GEMINI_API_KEY);
  assert(!calls[1].url.includes(env.GEMINI_API_KEY));
  assert.deepEqual(reports.filter(item=>item.code!=="ROUTE_SELECTED"), [{ provider: "groq", code: "PROVIDER_RATE_LIMIT" }]);
});

test("safety-blocked responses do not bypass the block through another provider", async () => {
  let calls = 0;
  await assert.rejects(generateStudy(input, { env: { ...env, AI_PROVIDER_ORDER: "gemini,groq" }, fetcher: async () => {
    calls++;
    return { ok: true, json: async () => ({ promptFeedback: { blockReason: "SAFETY" } }) };
  } }), error => error.code === "CONTENT_BLOCKED" && error.status === 422);
  assert.equal(calls, 1);
});

test("network failures fail over, while invalid requests stop without repeating the request", async () => {
  let calls = 0;
  const result = await generateStudy(input, { env, fetcher: async () => {
    if (++calls === 1) throw new Error("network details that must stay private");
    return { ok: true, json: async () => ({ candidates: [{ content: { parts: [{ text: "A real provider response" }] } }] }) };
  } });
  assert.equal(result.provider, "gemini");
  calls = 0;
  await assert.rejects(generateStudy(input, { env, fetcher: async () => {
    calls++; return { ok: false, status: 400, json: async () => ({}) };
  } }), error => error.code === "HTTP_400");
  assert.equal(calls, 1);
});

test("Gemini accepts inline recordings and requests verbatim transcription", async () => {
  const audio = { audio: "dGVzdA==", filename: "lecture.webm" };
  const payload = geminiPayload(audio);
  assert.deepEqual(payload.contents[0].parts[1], { inlineData: { mimeType: "audio/webm", data: audio.audio } });
  assert.match(payload.contents[0].parts[0].text, /verbatim/);
});

test("unconfigured providers never make a network request or invent an answer", async () => {
  await assert.rejects(generateStudy(input, { env: {}, fetcher: () => { throw new Error("Unexpected request"); } }),
    error => error.code === "NOT_CONFIGURED" && error.status === 503);
});

test("truncated course/JSON output is reported instead of presented as a complete answer", async () => {
  await assert.rejects(generateStudy(input, { env: { GEMINI_ENABLED: "true", GEMINI_API_KEY: "test", GEMINI_MODEL: "test-model" },
    fetcher: async () => ({ ok: true, json: async () => ({ candidates: [{ finishReason: "MAX_TOKENS" }] }) }) }),
  error => error.code === "OUTPUT_LIMIT");
});

test("uses the server alias and discovers an actually listed text model", async () => {
  const urls = [];
  const result = await generateStudy(input, { env: { Groq_Key: "fixture-alias" }, fetcher: async (url, options) => {
    urls.push(url); assert.equal(options.headers.Authorization, "Bearer fixture-alias");
    if (url.endsWith("/models")) return { ok: true, json: async () => ({ data: [{ id: "llama-3.3-70b-versatile", active: true }] }) };
    assert.equal(JSON.parse(options.body).model, "llama-3.3-70b-versatile");
    return { ok: true, json: async () => ({ choices: [{ message: { content: "A provider response." } }] }) };
  } });
  assert.equal(result.provider, "groq"); assert.equal(urls.length, 2); assert(urls.every(url => !url.includes("fixture-alias")));
});
test("a rejected credential is not repeatedly sent by the same warm server", async () => {
  let calls = 0;
  const options = { env: { Groq_Key: "fixture-rejected", GROQ_TEXT_MODEL: "fixture" }, fetcher: async () => { calls++; return { ok: false, status: 401, json: async () => ({}) }; } };
  await assert.rejects(generateStudy(input, options), error => error.code === "CREDENTIALS_REJECTED");
  await assert.rejects(generateStudy(input, options), error => error.code === "CREDENTIALS_REJECTED");
  assert.equal(calls, 1);
});

test("a busy Groq model falls back to a listed model without disabling transcription", async () => {
  const seen = [], options = { env: { Groq_Key: "fixture-model-fallback" }, fetcher: async (url, request) => {
    if (url.endsWith("/models")) return { ok: true, json: async () => ({ data: ["llama-3.1-8b-instant", "openai/gpt-oss-20b"].map(id => ({ id })) }) };
    if (url.endsWith("/transcriptions")) return { ok: true, json: async () => ({ text: "A recorded question." }) };
    const body = JSON.parse(request.body); seen.push(body.model);
    return body.model === "llama-3.1-8b-instant" ? { ok: false, status: 429, headers: { get: () => "12" }, json: async () => ({}) }
      : { ok: true, json: async () => ({ choices: [{ message: { content: "A real response from the available model." } }] }) };
  } };
  assert.equal((await generateStudy(input, options)).provider, "groq");
  assert.deepEqual(seen, ["llama-3.1-8b-instant", "openai/gpt-oss-20b"]);
  assert.equal((await generateStudy({ audio: "dGVzdA==", filename: "question.webm" }, options)).text, "A recorded question.");
  await generateStudy(input, options); assert.equal(seen.at(-1), "openai/gpt-oss-20b");
});

test("an oversized generation does not put a later small greeting into cooldown", async () => {
  const options = { env: { Groq_Key: "fixture-size" }, fetcher: async (url, request) => {
    if (url.endsWith("/models")) return { ok: true, json: async () => ({ data: ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"].map(id => ({ id })) }) };
    return JSON.parse(request.body).max_completion_tokens > 500 ? { ok: false, status: 429, json: async () => ({ error: { message: "Request too large for the model token allowance." } }) }
      : { ok: true, json: async () => ({ choices: [{ message: { content: "Hello!" } }] }) };
  } };
  await assert.rejects(generateStudy({ ...input, maxTokens: 5500 }, options), error => error.code === "REQUEST_TOO_LARGE");
  assert.equal((await generateStudy({ ...input, maxTokens: 200 }, options)).text, "Hello!");
});

test('two independently configured Gemini quota groups fail over sequentially without duplicate requests',async()=>{
  const keys=[],settings={GEMINI_ENABLED:'true',GEMINI_MODEL:'test',GEMINI_API_KEY:'first',GEMINI_API_KEY_2:'second',GEMINI_QUOTA_GROUP_1:'project-a',GEMINI_QUOTA_GROUP_2:'project-b'};
  const result=await generateStudy(input,{env:settings,fetcher:async(url,request)=>{keys.push(request.headers['x-goog-api-key']);return keys.length===1?{ok:false,status:429,headers:{get:()=> '12'},json:async()=>({})}:{ok:true,json:async()=>({candidates:[{content:{parts:[{text:'Second project answered.'}]}}]})};}});
  assert.equal(result.text,'Second project answered.');assert.deepEqual(keys,['first','second']);
});
test('Gemini keys with unknown/shared quota do not rotate around a rate limit',async()=>{
  const keys=[],settings={GEMINI_ENABLED:'true',GEMINI_MODEL:'test',GEMINI_API_KEY:'first',Gemini_Key_2:'second'};
  await assert.rejects(generateStudy(input,{env:settings,fetcher:async(url,request)=>{keys.push(request.headers['x-goog-api-key']);return{ok:false,status:429,headers:{get:()=> '20'},json:async()=>({})};}}),error=>error.status===429);
  assert.deepEqual(keys,['first']);
});
test('identical in-flight style requests share one provider call and validated tasks can be cached',async()=>{
  let calls=0;const options={env:{GROQ_API_KEY:'dedupe',GROQ_TEXT_MODEL:'test'},fetcher:async()=>{calls++;await new Promise(resolve=>setTimeout(resolve,10));return{ok:true,json:async()=>({choices:[{message:{content:'{"font":"sans"}'}}]})};}};
  const request={...input,task:'style'};await Promise.all([generateStudy(request,options),generateStudy(request,options)]);await generateStudy(request,options);assert.equal(calls,1);
});
