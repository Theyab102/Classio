import { saveWorkspace, changedFields } from './cloudWorkspace';
import { auth } from './firebase';
import { TextEncoder } from 'util';

// jsdom lacks this browser API; use Node's equivalent for byte-size checks.
beforeAll(() => { global.TextEncoder = TextEncoder; });

const mockCommit = jest.fn(async () => {}), mockSet = jest.fn();
jest.mock('./firebase', () => ({ auth: { currentUser: { uid: 'alice' } }, db: {}, storage: {} }));
jest.mock('firebase/firestore', () => ({
  collection: jest.fn(), doc: (_db, ...parts) => parts.join('/'), getDoc: jest.fn(), getDocs: jest.fn(), onSnapshot: jest.fn(), runTransaction: jest.fn(),
  serverTimestamp: () => 'server-time', deleteField: () => 'deleted-field',
  writeBatch: () => ({ set: mockSet, commit: mockCommit }),
}));
jest.mock('firebase/storage', () => ({ getBlob: jest.fn(), getMetadata: jest.fn(), ref: jest.fn(), uploadBytesResumable: jest.fn() }));

beforeEach(() => { auth.currentUser = { uid: 'alice' }; mockSet.mockClear(); mockCommit.mockClear(); });

test('missing cloud storage does not stop notes/metadata saving or pretend that original bytes uploaded', async () => {
  const previous = process.env.REACT_APP_FIREBASE_STORAGE_BUCKET;
  delete process.env.REACT_APP_FIREBASE_STORAGE_BUCKET;
  try {
    const original = new File(['Original material'], 'physics.txt', { type: 'text/plain' });
    const before = [{ id: 'school', name: 'School', files: [{ id: 'physics', name: 'physics.txt', notes: 'Before' }] }];
    const next = [{ ...before[0], files: [{ ...before[0].files[0], notes: 'Edited notes', _fileObj: original }] }];
    const result = await saveWorkspace('alice', before, next);
    expect(mockCommit).toHaveBeenCalledTimes(1);
    expect(mockSet).toHaveBeenCalledWith('users/alice/files/physics', expect.objectContaining({ notes: 'Edited notes' }), { merge: true });
    expect(result.folders[0].files[0]._fileObj).toBe(original);
    expect(result.folders[0].files[0].cloudFile).toBeUndefined();
    expect(result.uploadWarnings).toHaveLength(1);
  } finally { if (previous === undefined) delete process.env.REACT_APP_FIREBASE_STORAGE_BUCKET; else process.env.REACT_APP_FIREBASE_STORAGE_BUCKET = previous; }
});

test('workspace writes reject an account switch before any database mutation', async () => {
  auth.currentUser = { uid: 'bob' };
  await expect(saveWorkspace('alice', [], [])).rejects.toThrow('account changed');
  expect(mockSet).not.toHaveBeenCalled();
  expect(mockCommit).not.toHaveBeenCalled();
});

test('field patches leave unrelated remote fields intact', () => {
  expect(changedFields({ id: 'physics', name: 'Physics', notes: 'Before' }, { id: 'physics', name: 'Physics', notes: 'After' })).toEqual({ notes: 'After' });
});
