import { collection, doc, getDoc, getDocs, onSnapshot, runTransaction, serverTimestamp, writeBatch, deleteField } from 'firebase/firestore';
import { getBlob, getMetadata, ref, uploadBytesResumable } from 'firebase/storage';
import { auth, db, storage } from './firebase';

export const MAX_CLOUD_BYTES = 20 * 1024 * 1024;
const uploads = new Map();
const digests = new WeakMap();
const clean = value => JSON.parse(JSON.stringify(value, (key, item) => key === '_fileObj' || key === '_updatedAt' ? undefined : item));
const validId = id => typeof id === 'string' && /^[A-Za-z0-9_.-]{1,150}$/.test(id);
function owned(uid) {
  if (!uid || auth.currentUser?.uid !== uid) throw new Error('Your account changed. Sign in again before syncing.');
}
function itemRef(uid, kind, id) {
  if (!validId(id)) throw new Error('A file or folder has an invalid identifier. Export it before retrying.');
  return doc(db, 'users', uid, kind, id);
}

// A field patch preserves unrelated edits made on another device. Deletions use
// tombstones so legacy data remains recoverable and stale devices cannot revive it.
export function changedFields(previous = {}, next = {}) {
  const delta = {};
  for (const key of new Set([...Object.keys(previous), ...Object.keys(next)])) {
    if (key === '_updatedAt') continue;
    if (JSON.stringify(previous[key]) !== JSON.stringify(next[key])) delta[key] = key in next ? next[key] : undefined;
  }
  return delta;
}

export async function migrateWorkspace(uid) {
  owned(uid);
  const marker = doc(db, 'users', uid, 'sync', 'migration-v1');
  if ((await getDoc(marker)).exists()) return;
  const legacy = (await getDoc(doc(db, 'users', uid))).data()?.folders || [];
  // The legacy arrays are never removed. Transaction retries make migration
  // safe when a phone and tablet first open the upgraded app simultaneously.
  for (const folder of legacy) {
    const { files = [], ...metadata } = clean(folder);
    const entries = [[itemRef(uid, 'folders', folder.id), metadata],
      ...files.map(file => [itemRef(uid, 'files', file.id), { ...file, folderId: folder.id }])];
    for (const [target, data] of entries) {
      await runTransaction(db, async transaction => {
        if (!(await transaction.get(target)).exists()) transaction.set(target, { ...data, _updatedAt: serverTimestamp() });
      });
    }
  }
  await runTransaction(db, async transaction => {
    if (!(await transaction.get(marker)).exists()) transaction.set(marker, { completedAt: serverTimestamp() });
  });
}

function assemble(folderDocs, fileDocs) {
  const folders = folderDocs.filter(folder => !folder._deleted).map(folder => ({ ...folder, files: [] }));
  const byId = new Map(folders.map(folder => [folder.id, folder]));
  for (const file of fileDocs.filter(file => !file._deleted)) {
    const folder = byId.get(file.folderId);
    if (folder) folder.files.push(file);
  }
  return folders;
}
export async function loadWorkspace(uid) {
  owned(uid);
  const [folders, files] = await Promise.all(['folders', 'files'].map(kind => getDocs(collection(db, 'users', uid, kind))));
  return assemble(folders.docs.map(item => item.data()), files.docs.map(item => item.data()));
}
export function watchWorkspace(uid, onChange, onError) {
  const current = {};
  const unsubscribes = ['folders', 'files'].map(kind => onSnapshot(collection(db, 'users', uid, kind), snapshot => {
    if (auth.currentUser?.uid !== uid) return;
    current[kind] = snapshot.docs.map(item => item.data());
    if (current.folders && current.files) onChange(assemble(current.folders, current.files));
  }, onError));
  return () => unsubscribes.forEach(unsubscribe => unsubscribe());
}

async function uploadOriginal(uid, file, onProgress) {
  const blob = file._fileObj;
  if (!blob) return file;
  if (!process.env.REACT_APP_FIREBASE_STORAGE_BUCKET) throw new Error('Original files are saved on this device. The operator must configure Firebase Storage to enable file recall on other devices.');
  if (blob.size > MAX_CLOUD_BYTES) throw new Error(`${file.name}: cloud uploads are limited to 20 MB. The original is still on this device.`);
  if (!digests.has(blob)) digests.set(blob, blob.arrayBuffer().then(bytes => crypto.subtle.digest('SHA-256', bytes)).then(hash => [...new Uint8Array(hash)].map(value => value.toString(16).padStart(2, '0')).join('')));
  const digest = await digests.get(blob);
  const path = `users/${uid}/files/${file.id}/${digest}`;
  if (file.cloudFile?.path === path) return file;
  if (uploads.has(path)) return { ...file, cloudFile: await uploads.get(path) };
  const operation = (async () => {
    owned(uid);
    const target = ref(storage, path);
    try { await getMetadata(target); }
    catch (error) {
      if (error.code !== 'storage/object-not-found') throw error;
      await new Promise((resolve, reject) => {
        const task = uploadBytesResumable(target, blob, { contentType: blob.type || 'application/octet-stream', customMetadata: { owner: uid } });
        const timeout = setTimeout(() => task.cancel(), 120000);
        task.on('state_changed', snapshot => onProgress?.(`Uploading ${file.name}: ${Math.round(100 * snapshot.bytesTransferred / snapshot.totalBytes)}%`), error => { clearTimeout(timeout); reject(error); }, () => { clearTimeout(timeout); resolve(); });
      });
    }
    owned(uid);
    return { path, size: blob.size, type: blob.type || 'application/octet-stream', sha256: digest };
  })();
  uploads.set(path, operation);
  try { return { ...file, cloudFile: await operation }; }
  finally { uploads.delete(path); }
}

export async function recallOriginal(uid, file) {
  owned(uid);
  if (!validId(file.id) || !file.cloudFile?.path?.startsWith(`users/${uid}/files/${file.id}/`)) return null;
  const blob = await getBlob(ref(storage, file.cloudFile.path), MAX_CLOUD_BYTES);
  owned(uid);
  const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer()))].map(value => value.toString(16).padStart(2, '0')).join('');
  if (actual !== file.cloudFile.sha256) throw new Error('The downloaded file did not pass its integrity check. Please retry.');
  return new File([blob], file.name, { type: file.cloudFile.type });
}

function flatten(folders) {
  const folderMap = new Map(), fileMap = new Map();
  for (const folder of clean(folders)) {
    const { files = [], ...metadata } = folder;
    folderMap.set(folder.id, metadata);
    for (const file of files) fileMap.set(file.id, { ...file, folderId: folder.id });
  }
  return { folders: folderMap, files: fileMap };
}

export async function saveWorkspace(uid, before, next, onProgress) {
  owned(uid);
  const uploaded = [];
  const uploadWarnings = [];
  for (const folder of next) {
    const files = [];
    for (const file of folder.files || []) {
      try { files.push(await uploadOriginal(uid, file, onProgress)); }
      catch (error) {
        owned(uid);
        // A failed original upload must not prevent notes/courses/metadata from
        // syncing. Retain the original locally and report its separate status.
        files.push(file);
        uploadWarnings.push(error.code?.startsWith('storage/') ? `${file.name}: upload did not finish.` : error.message);
      }
    }
    uploaded.push({ ...folder, files });
  }
  const previous = flatten(before), updated = flatten(uploaded);
  let batch = writeBatch(db), count = 0;
  const commit = async () => { if (count) { owned(uid); await batch.commit(); batch = writeBatch(db); count = 0; } };
  for (const kind of ['folders', 'files']) {
    for (const [id, data] of updated[kind]) {
      const delta = changedFields(previous[kind].get(id), data);
      if (!Object.keys(delta).length) continue;
      if (new TextEncoder().encode(JSON.stringify(data)).length > 800000) throw new Error('This document is too large to sync. Export it and split it into smaller documents.');
      const patch = Object.fromEntries(Object.entries(delta).map(([key, value]) => [key, value === undefined ? deleteField() : value]));
      batch.set(itemRef(uid, kind, id), { ...patch, _updatedAt: serverTimestamp() }, { merge: true });
      if (++count >= 400) await commit();
    }
    for (const id of previous[kind].keys()) {
      if (updated[kind].has(id)) continue;
      batch.set(itemRef(uid, kind, id), { id, _deleted: true, _updatedAt: serverTimestamp() }, { merge: true });
      if (++count >= 400) await commit();
    }
  }
  await commit();
  return { folders: uploaded, uploadWarnings };
}

// Audio is a separate immutable object; it never replaces the learner's source file.
export async function savePodcastAudio(uid, jobId, blob) {
  if(!/^[a-f0-9]{64}$/.test(jobId))throw new Error('The podcast identifier is invalid.');
  const id='podcast_'+jobId;
  const value=await uploadOriginal(uid,{id,name:'Classio podcast',_fileObj:new File([blob],'classio-podcast',{type:blob.type})});
  return {storageId:id,cloudFile:value.cloudFile};
}
export async function recallPodcastAudio(uid, record) {
  if(!/^podcast_[a-f0-9]{64}$/.test(record?.storageId||''))return null;
  return recallOriginal(uid,{id:record.storageId,name:'classio-podcast',cloudFile:record.cloudFile});
}
export async function saveRecordingAudio(uid, digest, blob) {
  if(!/^[a-f0-9]{64}$/.test(digest))throw new Error('The recording identifier is invalid.');
  const id='voice_'+digest;
  const value=await uploadOriginal(uid,{id,name:'Voice recording',_fileObj:new File([blob],'voice-recording',{type:blob.type})});
  return {storageId:id,cloudFile:value.cloudFile};
}
export async function recallRecordingAudio(uid, record) {
  if(!/^voice_[a-f0-9]{64}$/.test(record?.storageId||''))return null;
  return recallOriginal(uid,{id:record.storageId,name:'voice-recording',cloudFile:record.cloudFile});
}
