import {auth,linkVerification,restoreSignedInAccount} from './firebase';
import {requestJson} from './apiClient';
jest.mock('firebase/app',()=>({getApps:jest.fn(()=>[]),initializeApp:jest.fn(()=>({})),getApp:jest.fn(()=>({}))}));
jest.mock('firebase/auth',()=>({browserLocalPersistence:{},getAuth:()=>({currentUser:{uid:'alice',getIdToken:async()=> 'test-only-account-token'},authStateReady:async()=>{}}),GoogleAuthProvider:jest.fn(()=>({setCustomParameters:jest.fn()})),setPersistence:jest.fn(async()=>{}),getRedirectResult:()=>new Promise(()=>{}),signInWithPopup:jest.fn(),signInWithRedirect:jest.fn()}));
jest.mock('firebase/firestore',()=>({getFirestore:jest.fn(()=>({}))}));
jest.mock('firebase/storage',()=>({getStorage:jest.fn(()=>({}))}));
jest.mock('./apiClient',()=>({requestJson:jest.fn()}));
beforeEach(()=>requestJson.mockReset());
test('an HTTP-success new-stage reply cannot masquerade as synced account verification',async()=>{
 requestJson.mockResolvedValue({stage:'new'});
 await expect(linkVerification()).rejects.toThrow('Account verification could not sync.');
 expect(requestJson.mock.calls[0][1].body).toEqual({action:'link'});
});
test('account verification accepts only the confirmed stage and uses a bounded private request',async()=>{
 requestJson.mockResolvedValue({stage:'accepted'});
 await expect(linkVerification()).resolves.toBeUndefined();
 expect(requestJson.mock.calls[0][1]).toMatchObject({timeout:7000,headers:{Authorization:'Bearer test-only-account-token'}});
});
test('a cached signed-in account restores without waiting on a stalled redirect result',async()=>{
 expect(await restoreSignedInAccount()).toBe(auth.currentUser);
});
