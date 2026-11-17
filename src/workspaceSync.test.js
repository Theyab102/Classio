import {TextEncoder} from 'util';
import {getDoc,setDoc,deleteDoc} from 'firebase/firestore';
import {recordParts,loadCloudRecord,saveCloudRecord,syncErrorMessage,mergeRestoredWorkspace,cloudSyncDiagnostic} from './workspaceSync';
jest.mock('firebase/firestore',()=>({doc:(_db,...path)=>path.join('/'),getDoc:jest.fn(),setDoc:jest.fn(),deleteDoc:jest.fn()}));
const records=new Map();
beforeEach(()=>{global.TextEncoder=TextEncoder;records.clear();jest.clearAllMocks();getDoc.mockImplementation(async path=>({metadata:{fromCache:false},data:()=>records.get(path)}));setDoc.mockImplementation(async(path,value)=>{records.set(path,value);});deleteDoc.mockImplementation(async path=>{records.delete(path);});});
test('large workspaces sync in bounded parts and restore Arabic and surrogate pairs exactly',async()=>{
  const data={folders:[{id:'arabic',files:[{id:'one',notes:'مرحبا 🌙 '.repeat(100000)}]}]};
  await saveCloudRecord({},'owner','workspace-v2',data);
  expect(records.get('users/owner/preferences/workspace-v2').parts.length).toBeGreaterThan(1);
  for(const value of records.values())expect(new TextEncoder().encode(JSON.stringify(value)).length).toBeLessThan(900000);
  expect(await loadCloudRecord({},'owner','workspace-v2')).toEqual(data);
});
test('a failed part never replaces the last readable manifest',async()=>{
  const original={folders:[{name:'Keep my work'}]};await saveCloudRecord({},'owner','workspace-v2',original);
  const manifest=records.get('users/owner/preferences/workspace-v2');
  setDoc.mockRejectedValueOnce(Object.assign(new Error('Offline'),{code:'unavailable'}));
  await expect(saveCloudRecord({},'owner','workspace-v2',{folders:[]})).rejects.toThrow('Offline');
  expect(records.get('users/owner/preferences/workspace-v2')).toBe(manifest);
  expect(await loadCloudRecord({},'owner','workspace-v2')).toEqual(original);
});
test('a successful replacement cleans old parts and has an explicit access-error explanation',async()=>{
  await saveCloudRecord({},'owner','assistant',{chats:[{title:'First'}]});
  const old=records.get('users/owner/preferences/assistant').parts;
  await saveCloudRecord({},'owner','assistant',{chats:[{title:'Second'}]});
  expect(old.every(id=>!records.has('users/owner/preferences/'+id))).toBe(true);
  expect(await loadCloudRecord({},'owner','assistant')).toEqual({chats:[{title:'Second'}]});
  expect(syncErrorMessage({code:'permission-denied'})).toContain('deployed Firebase rules');
});
test('small chunks preserve Unicode boundaries',()=>{const value={text:'ع🌞x'.repeat(50)};const parts=recordParts(value,19);expect(parts.every(part=>new TextEncoder().encode(part).length<=19)).toBe(true);expect(JSON.parse(parts.join(''))).toEqual(value);});
test('access denial identifies the exact operation and path without publishing parts, account IDs or tokens',async()=>{
  getDoc.mockRejectedValue(Object.assign(new Error('Denied'),{code:'permission-denied'}));
  let failure;try{await saveCloudRecord({app:{options:{projectId:'classio-4378f'}}},'private-owner','workspace-v2',{folders:[]});}catch(error){failure=error;}
  expect(setDoc).not.toHaveBeenCalled();expect(failure.syncContext).toEqual({project:'classio-4378f',operation:'read manifest before save',path:'users/{uid}/preferences/workspace-v2'});
  const details=cloudSyncDiagnostic(failure,{uid:'private-owner',providerData:[{providerId:'google.com'}]},'classio-4378f','google.com');
  expect(details.providerEvidence).toBe('refreshed ID token');expect(JSON.stringify(details)).not.toContain('private-owner');
});

test('reconnecting keeps new local edits without resurrecting deleted materials or folders',()=>{const remote=[{id:'keep',files:[{id:'deleted',notes:'old'},{id:'edited',notes:'old'}]},{id:'gone',files:[{id:'inside'}]}],local=[{id:'keep',files:[{id:'edited',notes:'new'},{id:'new'}]}];expect(mergeRestoredWorkspace(remote,local,{folders:['gone'],files:['deleted']})).toEqual(local);});
