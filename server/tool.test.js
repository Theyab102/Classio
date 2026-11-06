const { test } = require('node:test');
const assert = require('node:assert/strict');
const admin = require('./admin');
const agent = require('../api/agent');
const { TERMS_VERSION } = require('./access');

const unit = { title: 'Forces', lesson: 'A force changes motion. For example, a push accelerates a cart.', question: 'Which unit measures force?', options: ['Newton', 'Joule', 'Watt'], correct: 0, explanation: 'Force is measured in newtons.' };
function body() { return { tool: 'create_course', requestId: '12345678-1234-1234-1234-123456789abc', arguments: { fileId: 'physics', topic: 'Forces', difficulty: 'beginner', lessons: 2 } }; }
function database(context, failWrite = false) {
  const values = new Map([
    ['users/alice/access/verification', { stage: 'accepted', adult: true, version: TERMS_VERSION }],
    ['users/alice/files/physics', { id: 'physics', folderId: 'school', name: 'Physics', notes: 'Forces change motion.' }],
    ['users/bob/files/private', { id: 'private', notes: 'Private Bob content' }],
  ]);
  const reference = path => ({ path, get: async () => ({ data: () => values.get(path) }), set: async (data, options) => values.set(path, options?.merge ? { ...values.get(path), ...data } : data) });
  context.mock.method(admin, 'identity', () => ({ verifyIdToken: async () => ({ uid: 'alice', email_verified: true, firebase: { sign_in_provider: 'google.com' } }) }));
  context.mock.method(admin, 'database', () => ({ doc: reference, runTransaction: async operation => operation({ get: target => target.get(),
    set: (target, data) => { if (failWrite && data.status === 'complete') throw new Error('private backend configuration detail'); values.set(target.path, data); },
    update: (target, data) => { if (failWrite) throw new Error('private backend configuration detail'); values.set(target.path, { ...values.get(target.path), ...data }); },
  }) }));
  return values;
}
async function invoke(payload = body(), authorization = 'Bearer test-token') {
  const output = { headers: {} };
  await agent({ method: 'POST', headers: { 'content-type': 'application/json', ...(authorization ? { authorization } : {}) }, body: payload }, {
    setHeader(name, value) { output.headers[name] = value; }, status(status) { output.status = status; return this; }, json(value) { output.body = value; },
  });
  return output;
}

test('course tool calls the provider, validates lessons and acknowledges only a completed database write (mocked services)', async context => {
  const values = database(context);
  context.mock.method(global, 'fetch', async (_url, options) => {
    assert.match(JSON.parse(options.body).messages[1].content, /Forces change motion/);
    return { ok: true, json: async () => ({ choices: [{ message: { content: JSON.stringify([unit, unit]) } }] }) };
  });
  const result = await invoke();
  assert.equal(result.status, 200); assert.equal(result.body.saved, true);
  assert.equal(values.get('users/alice/files/physics').course.units.length, 2);
  assert.equal(values.get('users/bob/files/private').notes, 'Private Bob content');
});

test('duplicate course requests reuse a completed result, while changed arguments with the same ID are rejected', async context => {
  database(context);
  const fetch = context.mock.method(global, 'fetch', async () => ({ ok: true, json: async () => ({ choices: [{ message: { content: JSON.stringify([unit, unit]) } }] }) }));
  assert.equal((await invoke()).status, 200);
  assert.equal((await invoke()).status, 200);
  const changed = body(); changed.arguments.topic = 'A different topic';
  assert.equal((await invoke(changed)).status, 409);
  assert.equal(fetch.mock.callCount(), 1);
});

test('course tool rejects anonymous callers and cannot read a different account file', async context => {
  database(context);
  const fetch = context.mock.method(global, 'fetch', () => { throw new Error('Unexpected provider request'); });
  assert.equal((await invoke(body(), '')).status, 401);
  const request = body(); request.arguments.fileId = 'private';
  assert.equal((await invoke(request)).status, 404);
  assert.equal(fetch.mock.callCount(), 0);
});

test('a failed database write never returns saved:true or exposes backend exception text', async context => {
  database(context, true);
  context.mock.method(global, 'fetch', async () => ({ ok: true, json: async () => ({ choices: [{ message: { content: JSON.stringify([unit, unit]) } }] }) }));
  const result = await invoke();
  assert.equal(result.status, 503);
  assert.notEqual(result.body.saved, true);
  assert.doesNotMatch(JSON.stringify(result.body), /private backend/);
});
