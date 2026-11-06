const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { initializeTestEnvironment, assertFails, assertSucceeds } = require('@firebase/rules-unit-testing');
const { doc, setDoc, getDoc, updateDoc } = require('firebase/firestore');
const { ref, uploadBytes, getBytes } = require('firebase/storage');

let environment;
before(async () => {
  environment = await initializeTestEnvironment({ projectId: 'demo-classio',
    firestore: { host: '127.0.0.1', port: 8081, rules: fs.readFileSync('firestore.rules', 'utf8') },
    storage: { host: '127.0.0.1', port: 9199, rules: fs.readFileSync('storage.rules', 'utf8') },
  });
});
after(async () => { await environment?.cleanup(); });
function account(uid, email = `${uid}@example.test`) {
  return environment.authenticatedContext(uid, { email, email_verified: true, firebase: { sign_in_provider: 'google.com' } });
}
const revision = 'a'.repeat(64);

test('an owner uploads a private original and a separate device for the same UID recalls the bytes', async () => {
  const phone = account('alice');
  const tablet = account('alice');
  const path = `users/alice/files/physics/${revision}`;
  await assertSucceeds(uploadBytes(ref(phone.storage(), path), Buffer.from('Private physics notes'), { customMetadata: { owner: 'alice' } }));
  const content = await assertSucceeds(getBytes(ref(tablet.storage(), path)));
  assert.equal(Buffer.from(content).toString(), 'Private physics notes');
});

test('another account cannot download an original, even if it claims the same email', async () => {
  const intruder = account('bob', 'alice@example.test');
  await assertFails(getBytes(ref(intruder.storage(), `users/alice/files/physics/${revision}`)));
  await assertFails(getBytes(ref(environment.unauthenticatedContext().storage(), `users/alice/files/physics/${revision}`)));
});

test('a file reference and notes can be recalled and updated by a second device', async () => {
  const phone = account('alice').firestore(), tablet = account('alice').firestore();
  const path = 'users/alice/files/physics';
  await assertSucceeds(setDoc(doc(phone, path), { id: 'physics', folderId: 'school', notes: 'Forces and motion', cloudFile: { path: `users/alice/files/physics/${revision}` } }));
  assert.equal((await getDoc(doc(tablet, path))).data().notes, 'Forces and motion');
  await assertSucceeds(updateDoc(doc(tablet, path), { notes: 'Mass times acceleration' }));
  assert.equal((await getDoc(doc(phone, path))).data().notes, 'Mass times acceleration');
  await assertFails(getDoc(doc(account('bob').firestore(), path)));
});

test('account verification and server job records cannot be forged through the client SDK', async () => {
  const db = account('alice').firestore();
  await assertFails(setDoc(doc(db, 'users/alice/access/verification'), { stage: 'accepted' }));
  await assertFails(setDoc(doc(db, 'users/alice/aiJobs/forged'), { status: 'complete' }));
  await assertFails(setDoc(doc(db, 'aiRateLimits/study-all'), { count: 0 }));
});

test('profile and chat records are private to the account', async () => {
  const alice = account('alice').firestore(), bob = account('bob').firestore();
  for (const path of ['users/alice/profile/avatar', 'users/alice/conversations/test-chat']) {
    await assertSucceeds(setDoc(doc(alice, path), { value: 'Private account state' }));
    await assertFails(getDoc(doc(bob, path)));
    await assertFails(setDoc(doc(bob, path), { value: 'Overwrite' }));
  }
});

test('original versions are immutable and another user cannot upload into an owner path', async () => {
  const target = `users/alice/files/immutable/${'b'.repeat(64)}`;
  await assertSucceeds(uploadBytes(ref(account('alice').storage(), target), Buffer.from('First version'), { customMetadata: { owner: 'alice' } }));
  await assertFails(uploadBytes(ref(account('alice').storage(), target), Buffer.from('Replacement'), { customMetadata: { owner: 'alice' } }));
  await assertFails(uploadBytes(ref(account('bob').storage(), `users/alice/files/new/${'c'.repeat(64)}`), Buffer.from('Foreign file'), { customMetadata: { owner: 'alice' } }));
});
