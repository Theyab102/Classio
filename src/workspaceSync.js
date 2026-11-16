import {doc, getDoc, setDoc, deleteDoc} from 'firebase/firestore';
import {makeStudyId,mergeSessionFolders} from './studyTools';
import {withRestoreTimeout} from './authErrors';

export function recordParts(value, maxBytes = 580000) {
  const json = JSON.stringify(value);
  const encoder = new TextEncoder();
  if (encoder.encode(json).length > 30 * 1024 * 1024) throw Object.assign(new Error('This workspace is larger than the cloud sync limit. Export large presentations or remove unused media.'), {code:'workspace/too-large'});
  const parts = []; let chunk = '', size = 0;
  for (const char of json) {
    const point=char.codePointAt(0),bytes=point<=0x7f?1:point<=0x7ff?2:point<=0xffff?3:4;
    if (size + bytes > maxBytes) { parts.push(chunk); chunk = ''; size = 0; }
    chunk += char; size += bytes;
  }
  if (chunk) parts.push(chunk);
  return parts;
}
const ref = (db, uid, key) => doc(db, 'users', uid, 'preferences', key);
export async function loadCloudRecord(db, uid, key, fallback = null) {
  for (let attempt = 0; attempt < 2; attempt++) {
    const snapshot = await withRestoreTimeout(getDoc(ref(db, uid, key)));
    if (snapshot.metadata?.fromCache) throw Object.assign(new Error('Only an offline copy was available.'), {code:'unavailable'});
    const manifest = snapshot.data();
    if (manifest?.format !== 'classio-parts-v1') return manifest ?? fallback;
    if (!Array.isArray(manifest.parts) || !manifest.parts.length || manifest.parts.length > 100 || manifest.parts.some(id=>typeof id!=='string'||!id.startsWith(key+'-part-')||id.includes('/'))) throw new Error('The saved workspace index was incomplete.');
    try {
      const parts = [];
      for (let start = 0; start < manifest.parts.length; start += 4) {
        parts.push(...await Promise.all(manifest.parts.slice(start,start+4).map(async id=>{
          const part = await withRestoreTimeout(getDoc(ref(db,uid,id)));
          if (part.metadata?.fromCache || part.data()?.revision !== manifest.revision || typeof part.data()?.data !== 'string') throw new Error('A saved workspace part was unavailable.');
          return part.data().data;
        })));
      }
      return JSON.parse(parts.join(''));
    } catch (error) { if (attempt) throw error; }
  }
}
const queues = new Map();
export function saveCloudRecord(db, uid, key, value, isCurrent = () => true) {
  const queueKey = uid + ':' + key;
  const previous = queues.get(queueKey) || Promise.resolve();
  const job = previous.catch(()=>{}).then(async()=>{
    if (!isCurrent()) return;
    const parts = recordParts(value), revision = makeStudyId(), ids = parts.map((_,i)=>`${key}-part-${revision}-${i}`);
    let old;
    try { old = (await withRestoreTimeout(getDoc(ref(db,uid,key)))).data(); } catch {}
    for (let start = 0; start < parts.length; start += 4) {
      if (!isCurrent()) return;
      await Promise.all(parts.slice(start,start+4).map((data,i)=>withRestoreTimeout(setDoc(ref(db,uid,ids[start+i]),{data,revision}))));
    }
    if (!isCurrent()) { await Promise.allSettled(ids.map(id=>deleteDoc(ref(db,uid,id)))); return; }
    await withRestoreTimeout(setDoc(ref(db,uid,key),{format:'classio-parts-v1',revision,parts:ids,updatedAt:Date.now()}));
    if (old?.format==='classio-parts-v1' && Array.isArray(old.parts)) await Promise.allSettled(old.parts.filter(id=>typeof id==='string'&&id.startsWith(key+'-part-')&&!id.includes('/')).map(id=>deleteDoc(ref(db,uid,id))));
  });
  queues.set(queueKey,job);
  job.finally(()=>{if(queues.get(queueKey)===job)queues.delete(queueKey);}).catch(()=>{});
  return job;
}
export function syncErrorMessage(error) {
  if (/permission-denied|unauthenticated/.test(error?.code || '')) return 'Cloud access was denied. Sign in again; if it continues, the site owner must check the deployed Firebase rules.';
  if (/resource-exhausted/.test(error?.code || '')) return 'Cloud storage is temporarily at its quota. Your changes remain on this device.';
  return error?.code === 'workspace/too-large' ? error.message : 'Cloud sync is unavailable. Your changes remain on this device; retry when connected.';
}

// Local deletions made while restoration was offline must survive reconnection.
export function mergeRestoredWorkspace(remote,local,removed={}) {
  const folders=new Set(removed.folders||[]),files=new Set(removed.files||[]);
  return mergeSessionFolders(remote.filter(folder=>!folders.has(folder.id)).map(folder=>({...folder,files:(folder.files||[]).filter(file=>!files.has(file.id))})),local);
}
