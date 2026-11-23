const { test, beforeEach } = require("node:test"),
  assert = require("node:assert/strict");
const {
  generateStudy,
  clearProviderCaches,
  providerHealth,
} = require("./ai-providers");
const { verificationError } = require("./ai-auth");
const { accessDiagnostic } = require("./access-diagnostic");
const settings = {
  Groq_Key: "fixture-routing-primary",
  Groq_Key_packup: "fixture-routing-backup",
  AI_GROQ_STUDENT_USE_APPROVED: "true",
  AI_GROQ_FREE_TIER_CONFIRMED: "true",
};
const request = {
  messages: [{ role: "user", content: "Explain gravity." }],
  maxTokens: 500,
};
const response = (data, status = 200) => ({
  ok: status === 200,
  status,
  headers: { get: () => null },
  json: async () => data,
});
const answer = () =>
  response({ choices: [{ message: { content: "A complete response." } }] });
const catalog = () =>
  response({
    data: [
      { id: "llama-3.1-8b-instant", context_window: 131072 },
      { id: "llama-3.3-70b-versatile", context_window: 131072 },
      { id: "whisper-large-v3-turbo" },
      {
        id: "meta-llama/llama-4-scout-17b-16e-instruct",
        context_window: 131072,
      },
    ],
  });
beforeEach(clearProviderCaches);
test("short study requests use the study model while classification and conversation keep the fast model", async () => {
  const used = [];
  const options = {
    env: settings,
    fetcher: async (url, options) =>
      url.endsWith("/models")
        ? catalog()
        : (used.push(JSON.parse(options.body).model), answer()),
  };
  for (const task of [
    "classification",
    "notes",
    "questions",
    "flashcards",
    "presentation",
    "podcast",
    "conversation",
  ])
    await generateStudy({ ...request, task }, options);
  assert.deepEqual(used, [
    "llama-3.1-8b-instant",
    ...Array(5).fill("llama-3.3-70b-versatile"),
    "llama-3.1-8b-instant",
  ]);
});
test("text model pins cannot disable vision or transcription and task-specific overrides stay server-only", async () => {
  const env = {
      ...settings,
      AI_GROQ_MODEL: "llama-3.1-8b-instant",
      AI_GROQ_STUDY_MODEL: "llama-3.3-70b-versatile",
    },
    used = [];
  const options = {
    env,
    fetcher: async (url, options) => {
      if (url.endsWith("/models")) return catalog();
      used.push(
        options.body instanceof FormData
          ? options.body.get("model")
          : JSON.parse(options.body).model,
      );
      return url.endsWith("/transcriptions")
        ? response({ text: "Transcript" })
        : answer();
    },
  };
  await generateStudy({ ...request, task: "notes" }, options);
  await generateStudy({ ...request, vision: true }, options);
  await generateStudy({ audio: "dGVzdA==", filename: "speech.webm" }, options);
  assert.deepEqual(used, [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "whisper-large-v3-turbo",
  ]);
});
test("an audio rate limit never consumes a backup key or disables text generation", async () => {
  const calls = [],
    options = {
      env: settings,
      fetcher: async (url, options) => {
        if (url.endsWith("/models")) return catalog();
        calls.push(options.headers.Authorization);
        return url.endsWith("/transcriptions") ? response({}, 429) : answer();
      },
    };
  await assert.rejects(
    generateStudy({ audio: "dGVzdA==", filename: "speech.webm" }, options),
    (error) => error.code === "PROVIDER_RATE_LIMIT",
  );
  assert.equal(
    (await generateStudy(request, options)).text,
    "A complete response.",
  );
  assert.deepEqual(calls, [
    "Bearer fixture-routing-primary",
    "Bearer fixture-routing-primary",
  ]);
  assert.equal(providerHealth().groq.capabilities.audio, "temporarily-limited");
  assert.equal(providerHealth().groq.status, "available");
});
test("Cloudflare reads array context metadata, skips retired models, and selects a current study model", async () => {
  const env = {
    CLOUDFLARE_API_TOKEN: "fixture-cloudflare",
    CLOUDFLARE_ACCOUNT_ID: "a".repeat(32),
    AI_CLOUDFLARE_STUDENT_USE_APPROVED: "true",
    AI_CLOUDFLARE_FREE_TIER_CONFIRMED: "true",
  };
  let sent;
  const result = await generateStudy(
    {
      ...request,
      task: "notes",
      messages: [{ role: "user", content: "x".repeat(24000) }],
      maxTokens: 2000,
    },
    {
      env,
      fetcher: async (url, options) =>
        url.endsWith("/search")
          ? response({
              result: [
                {
                  name: "@cf/meta/llama-3.1-8b-instruct-fp8",
                  properties: [
                    {
                      property_id: "planned_deprecation_date",
                      value: "2020-01-01",
                    },
                  ],
                },
                {
                  name: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                  task: { name: "Text Generation" },
                  properties: [
                    { property_id: "context_window", value: "24000" },
                  ],
                },
              ],
            })
          : ((sent = url),
            response({ result: { response: "Long study answer" } })),
    },
  );
  assert.equal(result.text, "Long study answer");
  assert(sent.endsWith("/@cf/meta/llama-3.3-70b-instruct-fp8-fast"));
});
test("OpenRouter image-output models cannot be selected for text requests", async () => {
  await assert.rejects(
    generateStudy(request, {
      env: {
        OPENROUTER_API_KEY: "fixture-router",
        AI_OPENROUTER_STUDENT_USE_APPROVED: "true",
      },
      fetcher: async (url) => {
        assert(url.endsWith("/models"));
        return response({
          data: [
            {
              id: "fixture/image:free",
              architecture: {
                input_modalities: ["text"],
                output_modalities: ["image"],
              },
              pricing: { prompt: "0", completion: "0" },
            },
          ],
        });
      },
    }),
    (error) => error.code === "MODEL_UNAVAILABLE",
  );
});
test("authentication failures retain fixed phase/reason diagnostics and never raw SDK details", () => {
  for (const [error, reason, reference] of [
    [
      Object.assign(Error("private module path and credential"), {
        code: "MODULE_NOT_FOUND",
      }),
      "FIREBASE_SDK_LOAD_FAILED",
      "A101",
    ],
    [
      Object.assign(Error("private network response"), {
        code: "app/network-error",
      }),
      "FIREBASE_NETWORK_FAILED",
      "A202",
    ],
    [
      Object.assign(Error("private administrator response"), {
        code: "auth/insufficient-permission",
      }),
      "FIREBASE_ADMIN_CONFIGURATION_FAILED",
      "A301",
    ],
    [Error("unknown private token details"), "FIREBASE_VERIFICATION_FAILED", "A500"],
  ]) {
    const safe = verificationError(error, "app_initialization");
    assert.equal(safe.reason, reason);
    assert.deepEqual(accessDiagnostic(safe, safe.code), {
      code: safe.code,
      reason,
      phase: "app_initialization",
      reference,
    });
    assert(!JSON.stringify(safe).includes("private"));
  }
  assert.deepEqual(
    accessDiagnostic(
      { reason: "private credentials", phase: "private token" },
      "AUTH_UNAVAILABLE",
    ),
    { code: "AUTH_UNAVAILABLE" },
  );
});

test("Cloudflare searches later catalog pages before declaring a configured model unavailable", async () => {
  const env = {
    CLOUDFLARE_API_TOKEN: "fixture-cf-pages",
    CLOUDFLARE_ACCOUNT_ID: "a".repeat(32),
    AI_CLOUDFLARE_STUDENT_USE_APPROVED: "true",
    AI_CLOUDFLARE_FREE_TIER_CONFIRMED: "true",
  };
  const urls = [];
  const result = await generateStudy(request, {
    env,
    fetcher: async (url) => {
      urls.push(url);
      if (url.endsWith("/search"))
        return response({
          result: [],
          result_info: { total_pages: 2, per_page: 20 },
        });
      if (url.includes("/search?page=2"))
        return response({
          result: [
            {
              name: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
              properties: [{ property_id: "context_window", value: "24000" }],
            },
          ],
        });
      return response({ result: { response: "Catalog page two answer" } });
    },
  });
  assert.equal(result.text, "Catalog page two answer");
  assert.equal(urls.filter((url) => url.includes("/models/search")).length, 2);
});

test("model output limits are checked before inference rather than returning truncated study work", async () => {
  const models = [];
  await generateStudy(
    { ...request, maxTokens: 2000 },
    {
      env: settings,
      fetcher: async (url, options) => {
        if (url.endsWith("/models"))
          return response({
            data: [
              {
                id: "llama-3.3-70b-versatile",
                context_window: 131072,
                max_completion_tokens: 1000,
              },
              {
                id: "llama-3.1-8b-instant",
                context_window: 131072,
                max_completion_tokens: 8192,
              },
            ],
          });
        models.push(JSON.parse(options.body).model);
        return answer();
      },
    },
  );
  assert.deepEqual(models, ["llama-3.1-8b-instant"]);
});

test("Cloudflare model billing restrictions do not rotate backup tokens", async () => {
  const env = {
    CLOUDFLARE_API_TOKEN: "fixture-primary",
    CLOUDFLARE_API_TOKEN_packup: "fixture-backup",
    CLOUDFLARE_ACCOUNT_ID: "a".repeat(32),
    AI_CLOUDFLARE_STUDENT_USE_APPROVED: "true",
    AI_CLOUDFLARE_FREE_TIER_CONFIRMED: "true",
  };
  const seen = [];
  await assert.rejects(
    generateStudy(request, {
      env,
      fetcher: async (url, options) => {
        if (url.endsWith("/search"))
          return response({
            result: [
              {
                name: "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                properties: [{ property_id: "context_window", value: "24000" }],
              },
            ],
          });
        seen.push(options.headers.Authorization);
        return response(
          { errors: [{ code: 5035, message: "private billing details" }] },
          403,
        );
      },
    }),
    (error) => error.code === "MODEL_REQUIRES_PAID",
  );
  assert.deepEqual(seen, ["Bearer fixture-primary"]);
});

test('planning uses a fast model and a short algebra conversation uses a study model', async()=>{
 const models=[];const options={env:settings,fetcher:async(url,options)=>url.endsWith('/models')?catalog():(models.push(JSON.parse(options.body).model),answer())};
 await generateStudy({...request,task:'planning'},options);await generateStudy({...request,task:'conversation',messages:[{role:'user',content:'Solve this quadratic equation step by step: x^2-3*x+2=0.'}]},options);
 assert.deepEqual(models,['llama-3.1-8b-instant','llama-3.3-70b-versatile']);
});
test('a later failed provider does not hide a recoverable capacity reset or consume a backup key',async()=>{
 const calls=[];await assert.rejects(generateStudy(request,{env:{...settings,OPENROUTER_API_KEY:'fixture-router-fallback',AI_OPENROUTER_STUDENT_USE_APPROVED:'true'},fetcher:async(url,options)=>{
  if(url.includes('groq')&&url.endsWith('/models'))return catalog();
  if(url.includes('groq')){calls.push(options.headers.Authorization);return {...response({},429),headers:{get:()=> '86400'}};}
  return response({},503);
 }}),error=>error.code==='PROVIDER_RATE_LIMIT'&&error.retryAfter===86400);
 assert.deepEqual(calls,['Bearer fixture-routing-primary']);
});
test('small planning jobs prefer Cloudflare and teaching jobs prefer Groq when both are eligible',async()=>{
 const calls=[],env={...settings,CLOUDFLARE_API_TOKEN:'fixture-cf-lanes',CLOUDFLARE_ACCOUNT_ID:'a'.repeat(32),AI_CLOUDFLARE_STUDENT_USE_APPROVED:'true',AI_CLOUDFLARE_FREE_TIER_CONFIRMED:'true'};
 const options={env,fetcher:async(url,options)=>{
  if(url.endsWith('/search'))return response({result:[{name:'@cf/meta/llama-3.1-8b-instruct',properties:{context_window:131072}}]});
  if(url.endsWith('/models'))return catalog();calls.push(url.includes('cloudflare')?'cloudflare':'groq');return url.includes('cloudflare')?response({result:{response:'A complete response.'}}):answer();
 }};
 await generateStudy({...request,task:'planning'},options);await generateStudy({...request,task:'notes'},options);assert.deepEqual(calls,['cloudflare','groq']);
});
