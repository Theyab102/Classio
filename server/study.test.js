const { beforeEach, test } = require("node:test");
const assert = require("node:assert/strict");

let requestQuota;
let quotaReady;
const quotaPath = require.resolve("./ai-quota");
require.cache[quotaPath] = {
  exports: {
    configured: () => quotaReady,
    reserveRequest: async (_req, kind) => { requestQuota.kind = kind; return requestQuota; },
  },
};
const study = require("../api/study");
const { parseAudioRequest } = require("./study-request");
const { setAccess } = require('./access');

beforeEach(() => {
  requestQuota = {};
  quotaReady = true;
  process.env.AI_RATE_LIMIT_SECRET = 'classio-test-signing-secret-over-32-characters';
  process.env.GROQ_API_KEY = 'test-only-groq-key';
});

const messages = [{ role: "user", content: "Explain gravity." }];
const headers = { "content-type": "application/json", "sec-fetch-site": "same-origin" };

async function invoke(overrides = {}) {
  const result = { headers: {} };
  setAccess({ setHeader(_name, value) { result.cookie = value.split(';')[0]; } }, { stage: 'accepted', adult: true });
  await study({ method: "POST", headers: { ...headers, cookie: result.cookie }, body: { messages, maxTokens: 500 }, ...overrides }, {
    setHeader(name, value) { result.headers[name] = value; },
    status(code) { result.status = code; return this; },
    json(body) { result.body = body; },
  });
  return result;
}

test("a guest can generate a response without an authorization header", async context => {
  context.mock.method(global, "fetch", async (_url, options) => {
    assert.deepEqual(JSON.parse(options.body).messages, messages);
    return { ok: true, json: async () => ({ choices: [{ message: { content: "Gravity is attractive." } }] }) };
  });
  const reply = await invoke();
  assert.equal(reply.status, 200);
  assert.equal(reply.body.text, "Gravity is attractive.");
  assert.equal(reply.headers["Cache-Control"], "no-store");
  assert.equal(requestQuota.kind, "study");
});

test("audio uses the transcription provider and its own quota", async context => {
  context.mock.method(global, "fetch", async (url, options) => {
    assert.match(url, /audio\/transcriptions$/);
    assert.equal(options.body.get("file").name, "lecture.webm");
    return { ok: true, json: async () => ({ text: "Lecture transcript." }) };
  });
  const reply = await invoke({ body: { kind: "audio", audio: "dGVzdA==", filename: "lecture.webm" } });
  assert.equal(reply.status, 200);
  assert.equal(reply.body.text, "Lecture transcript.");
  assert.equal(requestQuota.kind, "audio");
  assert.equal(parseAudioRequest({ audio: "dGVzdA==", filename: "../lecture.webm" }), null);
  assert.equal(parseAudioRequest({ audio: "a".repeat(3800004), filename: "lecture.webm" }), null);
});

test("invalid and over-limit requests cannot reach an AI provider", async context => {
  const upstream = context.mock.method(global, "fetch", () => { throw new Error("Unexpected upstream call"); });
  assert.equal((await invoke({ method: "DELETE" })).status, 405);
  assert.equal((await invoke({ headers: { ...headers, "sec-fetch-site": "cross-site" } })).status, 403);
  assert.equal((await invoke({ headers: {} })).status, 415);
  assert.equal((await invoke({ body: {} })).status, 400);
  quotaReady = false;
  assert.equal((await invoke()).status, 503);
  quotaReady = true;
  requestQuota = { retryAfter: 60 };
  const limited = await invoke();
  assert.equal(limited.status, 429);
  assert.equal(limited.headers["Retry-After"], "60");
  assert.equal(upstream.mock.callCount(), 0);
});

test('missing or forged verification cannot consume AI quota or contact a provider', async context => {
  const upstream = context.mock.method(global, 'fetch', () => { throw new Error('Unexpected upstream call'); });
  const missing = await invoke({ headers });
  assert.equal(missing.status, 403);
  assert.equal(missing.body.code, 'VERIFICATION_REQUIRED');
  assert.equal((await invoke({ headers: { ...headers, cookie: 'classio_access_v2=accepted.forged' } })).status, 403);
  assert.equal(upstream.mock.callCount(), 0);
});

test('a Gemini preference cannot silently send audio to Groq', async context => {
  const upstream = context.mock.method(global, 'fetch', () => { throw new Error('Unexpected upstream call'); });
  const reply = await invoke({ body: { kind: 'audio', provider: 'gemini', audio: 'dGVzdA==', filename: 'lecture.webm' } });
  assert.equal(reply.status, 400);
  assert.equal(reply.body.code, 'AUDIO_PROVIDER_UNSUPPORTED');
  assert.equal(upstream.mock.callCount(), 0);
});
