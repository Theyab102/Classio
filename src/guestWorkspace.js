// Metadata and the encryption key belong to this tab's browsing session.
// Temporary file bytes are encrypted at rest; a new session cannot open them.
const SESSION='classio_guest_session_v1', WORKSPACE='classio_guest_workspace_v1';
let secret, database, writes=new Map(),leaseTimer,unlockLease,leaseRequest;
const encode=bytes=>btoa(String.fromCharCode(...bytes));
function session(){
  let saved=JSON.parse(sessionStorage.getItem(SESSION)||'null');
  if(!saved){saved={id:crypto.randomUUID(),key:encode(crypto.getRandomValues(new Uint8Array(32)))};sessionStorage.setItem(SESSION,JSON.stringify(saved));}
  return saved;
}
async function key(){if(!secret)secret=crypto.subtle.importKey('raw',Uint8Array.from(atob(session().key),c=>c.charCodeAt(0)),{name:'AES-GCM'},false,['encrypt','decrypt']);return secret;}
function open(){
  if(!database)database=new Promise((resolve,reject)=>{const request=indexedDB.open('classio_guest_files',1);request.onupgradeneeded=()=>request.result.createObjectStore('files');request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(new Error('Temporary file storage is unavailable.'));});
  return database;
}
async function transact(mode,run){const db=await open();return new Promise((resolve,reject)=>{const tx=db.transaction('files',mode),result=run(tx.objectStore('files'));tx.oncomplete=()=>resolve(result?.result);tx.onabort=tx.onerror=()=>reject(new Error('Temporary file storage is unavailable.'));});}
export function readGuestWorkspace(){try{return JSON.parse(sessionStorage.getItem(WORKSPACE)||'null');}catch{return null;}}
export function saveGuestMetadata(folders,chats,character){
  session();sessionStorage.setItem(WORKSPACE,JSON.stringify({folders:folders.map(folder=>({...folder,files:folder.files.map(({_fileObj,...file})=>file)})),chats,character}));
}
export async function saveGuestBlob(id,blob){
  if(!(blob instanceof Blob))return;
  if(writes.get(id)?.blob===blob)return writes.get(id).promise;
  const owner=session().id;
  const promise=(async()=>{const iv=crypto.getRandomValues(new Uint8Array(12)),header=JSON.stringify({name:blob.name||'',type:blob.type})+'\n',data=await crypto.subtle.encrypt({name:'AES-GCM',iv},await key(),await new Blob([header,blob]).arrayBuffer());await transact('readwrite',store=>store.put({iv,data,at:Date.now()},owner+':'+id));})();
  writes.set(id,{blob,promise});try{await promise;}catch(error){if(writes.get(id)?.promise===promise)writes.delete(id);throw error;}
}
export async function getGuestBlob(id){
  const pending=writes.get(id);if(pending)return pending.blob;
  const record=await transact('readonly',store=>store.get(session().id+':'+id));if(!record)return null;
  const data=new Uint8Array(await crypto.subtle.decrypt({name:'AES-GCM',iv:record.iv},await key(),record.data)),end=data.indexOf(10),meta=JSON.parse(new TextDecoder().decode(data.slice(0,end)));
  return meta.name?new File([data.slice(end+1)],meta.name,{type:meta.type}):new Blob([data.slice(end+1)],{type:meta.type});
}
export async function removeGuestBlob(id){const pending=writes.get(id);writes.delete(id);if(pending)await pending.promise.catch(()=>{});await transact('readwrite',store=>store.delete(session().id+':'+id));}
export function clearGuestWorkspace(){
  let owner;try{owner=JSON.parse(sessionStorage.getItem(SESSION)||'null')?.id;sessionStorage.removeItem(WORKSPACE);sessionStorage.removeItem(SESSION);}catch{}
  secret=null;writes=new Map();clearInterval(leaseTimer);unlockLease?.();leaseRequest?.abort();leaseRequest=null;unlockLease=null;
  if(owner)transact('readwrite',store=>{const cursor=store.openCursor();cursor.onsuccess=()=>{const item=cursor.result;if(item){if(String(item.key).startsWith(owner+':')||item.key==='lease:'+owner)item.delete();item.continue();}};}).catch(()=>{});
}
export async function pruneGuestFiles(){
  const current=session().id;
  // A lock protects other open tabs, including tabs suspended for a long time.
  // Without Web Locks, keep inaccessible ciphertext until explicit cleanup.
  if(!navigator.locks)return;
  if(!leaseRequest){leaseRequest=new AbortController();const signal=leaseRequest.signal;await new Promise(resolve=>{navigator.locks.request('classio-guest:'+current,{signal},async()=>{resolve();await new Promise(unlock=>unlockLease=unlock);}).catch(resolve);});if(signal.aborted)return;}
  const touch=()=>transact('readwrite',store=>store.put({at:Date.now()},'lease:'+current)).catch(()=>{});
  await touch();clearInterval(leaseTimer);leaseTimer=setInterval(touch,60000);
  const records=await transact('readonly',store=>store.getAllKeys()),owners=new Set(records.filter(id=>typeof id==='string'&&!id.startsWith('lease:')).map(id=>id.split(':')[0]));
  for(const owner of owners){
    if(owner===current)continue;
    const lease=await transact('readonly',store=>store.get('lease:'+owner));if(lease?.at>Date.now()-86400000)continue;
    await navigator.locks.request('classio-guest:'+owner,{ifAvailable:true},async lock=>{if(!lock)return;await transact('readwrite',store=>{const cursor=store.openCursor();cursor.onsuccess=()=>{const item=cursor.result;if(item){if((String(item.key).startsWith(owner+':')&&item.value.at<Date.now()-86400000)||item.key==='lease:'+owner)item.delete();item.continue();}};});});
  }
}
