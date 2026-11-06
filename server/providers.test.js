const { test, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert/strict');
const { generate: generateForEligibility, transcribe, geminiContent } = require('./ai-providers');
const { parseCourse, courseInput } = require('../src/shared/course-schema.js');
const generate=(input,selection,eligibility={adult:true})=>generateForEligibility(input,selection,eligibility);
const names = ['GROQ_API_KEY', 'GEMINI_API_KEY', 'GEMINI_MODEL', 'GEMINI_ENABLED', 'GEMINI_FREE_TIER_VERIFIED', 'GROQ_VISION_MODEL'];
let previous;
beforeEach(() => { previous = Object.fromEntries(names.map(name => [name, process.env[name]])); for (const name of names) delete process.env[name]; });
afterEach(() => { for (const name of names) { if (previous[name] === undefined) delete process.env[name]; else process.env[name] = previous[name]; } });
const input = { messages: [{ role: 'system', content: 'Be a careful tutor.' }, { role: 'user', content: 'Explain momentum.' }], maxTokens: 500 };
function enableBoth() { Object.assign(process.env, { GROQ_API_KEY: 'test-only-groq-key', GEMINI_API_KEY: 'test-only-gemini-key', GEMINI_MODEL: 'configured-test-model', GEMINI_ENABLED: 'true', GEMINI_FREE_TIER_VERIFIED: 'true' }); }

test('Groq uses its server credential and returns only generated text and routing metadata (mocked upstream)', async context => {
  enableBoth();
  context.mock.method(global, 'fetch', async (url, options) => {
    assert.equal(url, 'https://api.groq.com/openai/v1/chat/completions');
    assert.equal(options.headers.Authorization, 'Bearer test-only-groq-key');
    assert.deepEqual(JSON.parse(options.body).messages, input.messages);
    return { ok: true, json: async () => ({ choices: [{ message: { content: 'Momentum is mass times velocity.' } }] }) };
  });
  const output = await generate(input, { provider: 'groq' });
  assert.deepEqual(output, { text: 'Momentum is mass times velocity.', provider: 'groq', fallback: false });
  assert.doesNotMatch(JSON.stringify(output), /test-only/);
});

test('Gemini uses the official REST format and a header instead of a query-string key (mocked upstream)', async context => {
  enableBoth();
  context.mock.method(global, 'fetch', async (url, options) => {
    assert.equal(url, 'https://generativelanguage.googleapis.com/v1beta/models/configured-test-model:generateContent');
    assert.equal(options.headers['x-goog-api-key'], 'test-only-gemini-key');
    const body = JSON.parse(options.body);
    assert.equal(body.systemInstruction.parts[0].text, 'Be a careful tutor.');
    assert.equal(body.contents[0].role, 'user');
    return { ok: true, json: async () => ({ candidates: [{ content: { parts: [{ text: 'Hidden reasoning', thought: true }, { text: 'Momentum depends on mass and velocity.' }] } }] }) };
  });
  assert.equal((await generate(input, { provider: 'gemini' })).text, 'Momentum depends on mass and velocity.');
});

test('Auto prefers Groq and does not retry another provider without consent', async context => {
  enableBoth();
  const fetch = context.mock.method(global, 'fetch', async () => ({ ok: false, status: 429 }));
  await assert.rejects(generate(input, { provider: 'auto' }), error => error.status === 429 && error.code === 'PROVIDER_RATE_LIMIT');
  assert.equal(fetch.mock.callCount(), 1);
});

test('permitted temporary fallback makes exactly one alternate attempt and reports it', async context => {
  enableBoth();
  const fetch = context.mock.method(global, 'fetch', async url => url.includes('api.groq.com') ? { ok: false, status: 429 } : { ok: true, json: async () => ({ candidates: [{ content: { parts: [{ text: 'An explanation.' }] } }] }) });
  assert.deepEqual(await generate(input, { provider: 'auto', allowFallback: true }), { text: 'An explanation.', provider: 'gemini', fallback: true });
  assert.equal(fetch.mock.callCount(), 2);
});

test('quota exhaustion on both providers stops after two calls', async context => {
  enableBoth();
  const fetch = context.mock.method(global, 'fetch', async () => ({ ok: false, status: 429 }));
  await assert.rejects(generate(input, { provider: 'auto', allowFallback: true }), error => error.status === 429);
  assert.equal(fetch.mock.callCount(), 2);
});

test('invalid keys and unavailable models return safe errors without fallback or provider error-body leakage', async context => {
  enableBoth();
  const fetch = context.mock.method(global, 'fetch', async () => ({ ok: false, status: 401, text: () => 'private upstream error includes a credential' }));
  await assert.rejects(generate(input, { allowFallback: true }), error => error.code === 'PROVIDER_AUTH_FAILED' && !/credential|private upstream/.test(error.message));
  assert.equal(fetch.mock.callCount(), 1);
});

test('Gemini remains disabled until an operator explicitly configures an eligible model and free project', async context => {
  process.env.GEMINI_API_KEY = 'test-only-gemini-key';
  const fetch = context.mock.method(global, 'fetch', () => { throw new Error('Unexpected call'); });
  await assert.rejects(generate(input, { provider: 'gemini' }), error => error.code === 'PROVIDER_NOT_CONFIGURED');
  assert.equal(fetch.mock.callCount(), 0);
});

test('an unavailable model has a recoverable configuration error and never silently switches provider', async context => {
  enableBoth();
  const fetch = context.mock.method(global, 'fetch', async () => ({ ok: false, status: 404 }));
  await assert.rejects(generate(input, { provider: 'groq', allowFallback: true }), error => error.code === 'PROVIDER_MODEL_UNAVAILABLE');
  assert.equal(fetch.mock.callCount(), 1);
});

test('Gemini image conversion never accepts or fetches a remote image URL', () => {
  const body = geminiContent([{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,dGVzdA==' } }, { type: 'text', text: 'Explain' }] }]);
  assert.deepEqual(body.contents[0].parts[0], { inlineData: { mimeType: 'image/png', data: 'dGVzdA==' } });
});

test('an upstream reply cannot echo a known server credential to the browser', async context => {
  enableBoth();
  context.mock.method(global, 'fetch', async () => ({ ok: true, json: async () => ({ choices: [{ message: { content: 'Unsafe echo: test-only-groq-key and test-only-gemini-key' } }] }) }));
  const output = await generate(input, { provider: 'groq' });
  assert.doesNotMatch(output.text, /test-only-groq-key|test-only-gemini-key/);
  assert.match(output.text, /redacted credential/);
});

test('course tool schema rejects arbitrary tools, database paths, invalid provider selection and oversized arguments', () => {
  const valid = { tool: 'create_course', requestId: '12345678-1234-1234-1234-123456789abc', arguments: { topic: 'Motion', difficulty: 'beginner', lessons: 3, fileId: 'blank_123' } };
  assert.ok(courseInput(valid));
  assert.equal(courseInput({ ...valid, tool: 'delete_user' }), null);
  assert.equal(courseInput({ ...valid, arguments: { ...valid.arguments, fileId: '../other-user/private' } }), null);
  assert.equal(courseInput({ ...valid, arguments: { ...valid.arguments, lessons: 100 } }), null);
  assert.equal(courseInput({ ...valid, provider: 'arbitrary' }), null);
});

test('partial courses never pass validation or claim a saved result', () => {
  const unit = { title: 'Motion', lesson: 'Forces change motion.', question: 'A unit of force?', options: ['Newton', 'Watt', 'Joule'], correct: 0, explanation: 'A newton measures force.' };
  assert.equal(parseCourse(JSON.stringify([unit, unit]), 2).length, 2);
  assert.throws(() => parseCourse(JSON.stringify([unit, unit]), 3));
  assert.throws(() => parseCourse(JSON.stringify([unit, { ...unit, correct: 9 }])));
});


test('minor or unknown sessions cannot use Gemini, including permitted fallback', async context => {
  enableBoth();
  const fetch = context.mock.method(global,'fetch',async()=>({ok:false,status:429}));
  await assert.rejects(generate(input,{provider:'gemini'},{adult:false}),error=>error.code==='PROVIDER_AGE_RESTRICTED');
  assert.equal(fetch.mock.callCount(),0);
  await assert.rejects(generate(input,{provider:'groq',allowFallback:true},{adult:false}),error=>error.code==='PROVIDER_RATE_LIMIT');
  assert.equal(fetch.mock.callCount(),1);
});
