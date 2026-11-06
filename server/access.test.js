const { test, beforeEach } = require('node:test');
const assert = require('node:assert/strict');
const { setAccess, readAccess, authenticatedUser, requireAccess, TERMS_VERSION } = require('./access');
const admin = require('./admin');
const access = require('../api/access');

beforeEach(() => { process.env.CLASSIO_SESSION_SECRET = 'test-only-session-signing-secret-at-least-32-characters'; });
function cookie(claim) { let result; setAccess({ setHeader(_name, value) { result = value; } }, claim); return result; }
async function invoke(body, header = '') {
  const output = { headers: {} };
  await access({ method: body ? 'POST' : 'GET', body, headers: { 'content-type': 'application/json', cookie: header } }, {
    setHeader(name, value) { output.headers[name] = value; }, status(code) { output.status = code; return this; }, json(value) { output.body = value; },
  });
  return output;
}

test('access decisions are signed, HttpOnly, persistent, and contain no date of birth', () => {
  const signed = cookie({ stage: 'accepted', adult: true });
  assert.match(signed, /HttpOnly/); assert.match(signed, /Max-Age=31536000/); assert.match(signed, /SameSite=Lax/);
  assert.equal(readAccess({ headers: { cookie: signed } }).stage, 'accepted');
  const body = JSON.parse(Buffer.from(signed.split('=')[1].split('.')[0], 'base64url'));
  assert.equal(body.birthDate, undefined);
  assert.equal(body.version, TERMS_VERSION);
});

test('editing an accepted cookie or a local-storage flag cannot grant access', () => {
  const signed = cookie({ stage: 'blocked', adult: false });
  const forgedBody = Buffer.from(JSON.stringify({ stage: 'accepted', adult: true, version: TERMS_VERSION, exp: Date.now() + 10000 })).toString('base64url');
  assert.equal(readAccess({ headers: { cookie: signed.replace(/=[^.]+\./, '=' + forgedBody + '.') } }), null);
  assert.equal(readAccess({ headers: { cookie: 'classio_access_v1=accepted' } }), null);
});

test('server checks the exact minimum age and refuses impossible/future dates', () => {
  const today = new Date(2026, 10, 6);
  assert.deepEqual(access.ageDecision('2013-11-06', today), { stage: 'eligible', adult: false });
  assert.deepEqual(access.ageDecision('2013-11-07', today), { stage: 'blocked', adult: false });
  assert.equal(access.ageDecision('2026-02-30', today), null);
  assert.equal(access.ageDecision('2030-01-01', today), null);
});

test('terms cannot be accepted without the signed eligibility step', async () => {
  assert.equal((await invoke({ action: 'accept', termsAccepted: true })).status, 400);
  const screened = await invoke({ action: 'screen', birthDate: '2000-01-01' });
  assert.equal(screened.body.stage, 'eligible');
  const accepted = await invoke({ action: 'accept', termsAccepted: true }, screened.headers['Set-Cookie']);
  assert.equal(accepted.body.stage, 'accepted');
  assert.equal((await invoke(null, accepted.headers['Set-Cookie'])).body.stage, 'accepted');
});

test('blocked browser cannot immediately overwrite its decision by resubmitting another date', async () => {
  const blocked = await invoke({ action: 'screen', birthDate: '2020-01-01' });
  assert.equal(blocked.body.stage, 'blocked');
  assert.equal((await invoke({ action: 'screen', birthDate: '2000-01-01' }, blocked.headers['Set-Cookie'])).status, 403);
});

test('protected account tools reject unsigned callers and invalid identity tokens', async context => {
  await assert.rejects(authenticatedUser({ headers: {} }, true), error => error.status === 401);
  context.mock.method(admin, 'identity', () => ({ verifyIdToken: async () => { throw new Error('private credential detail'); } }));
  await assert.rejects(authenticatedUser({ headers: { authorization: 'Bearer invalid' } }, true), error => error.status === 401 && !/private/.test(error.message));
});

test('verified identity scopes account access to the token UID, ignoring caller-supplied user IDs', async context => {
  context.mock.method(admin, 'identity', () => ({ verifyIdToken: async () => ({ uid: 'alice', email_verified: true, firebase: { sign_in_provider: 'google.com' } }) }));
  const paths = [];
  context.mock.method(admin, 'database', () => ({ doc: path => { paths.push(path); return { get: async () => ({ data: () => ({ stage: 'accepted', adult: true, version: TERMS_VERSION }) }) }; } }));
  const result = await requireAccess({ headers: { authorization: 'Bearer signed-token' }, body: { uid: 'bob' } }, { account: true });
  assert.equal(result.user.uid, 'alice');
  assert.deepEqual(paths, ['users/alice/access/verification']);
});

test('account verification restores to a fresh browser without storing a birth date', async context => {
  context.mock.method(admin, 'identity', () => ({ verifyIdToken: async () => ({ uid: 'alice', email_verified: true, firebase: { sign_in_provider: 'google.com' } }) }));
  context.mock.method(admin, 'database', () => ({ doc: () => ({ get: async () => ({ data: () => ({ stage: 'accepted', adult: true, version: TERMS_VERSION }) }) }) }));
  const result = { headers: {} };
  await access({ method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer signed-token' }, body: { action: 'restore' } }, { setHeader(name, value) { result.headers[name] = value; }, status(status) { result.status = status; return this; }, json(body) { result.body = body; } });
  assert.equal(result.status, 200);
  assert.equal(readAccess({ headers: { cookie: result.headers['Set-Cookie'] } }).stage, 'accepted');
});
