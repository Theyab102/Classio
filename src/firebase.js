import { getApp, getApps, initializeApp } from 'firebase/app';
import { browserLocalPersistence, getAuth, GoogleAuthProvider, setPersistence, signInWithPopup, signInWithRedirect, getRedirectResult } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

// Web configuration identifies a public Firebase project; it is not an Admin key.
const app = getApps().length ? getApp() : initializeApp({
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY || 'AIzaSyCDDXRAxLwNmzSYxs0HnnjV2TfhbqSdiag',
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || 'classio-4378f.firebaseapp.com',
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || 'classio-4378f',
  appId: process.env.REACT_APP_FIREBASE_APP_ID || '1:595968221954:web:cdefee80f05999f8bf181b',
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || '595968221954',
  ...(process.env.REACT_APP_FIREBASE_STORAGE_BUCKET ? { storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET } : {}),
});
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);
const provider = new GoogleAuthProvider();
provider.setCustomParameters({ prompt: 'select_account' });
const persistence = setPersistence(auth, browserLocalPersistence);
export const authReady = persistence.then(() => getRedirectResult(auth));
authReady.catch(() => {});

export function authError(error) {
  const messages = {
    'auth/popup-closed-by-user': 'Sign-in was cancelled. Your work is still open.',
    'auth/cancelled-popup-request': 'Another sign-in window is already open.',
    'auth/unauthorized-domain': 'This domain is not authorized for Google sign-in. The operator must add it in Firebase Authentication.',
    'auth/account-exists-with-different-credential': 'This email already has an account with another sign-in method. Use that method before linking Google.',
    'auth/network-request-failed': 'Google sign-in could not connect. Check your connection and retry.',
  };
  return messages[error?.code] || 'Google sign-in could not finish. Please retry. Your work is still open.';
}
export async function signInGoogle() {
  try{localStorage.setItem('classio_google_signin_requested','true');}catch{}
  await persistence;
  try { return await signInWithPopup(auth, provider); }
  catch (error) {
    if (error.code === 'auth/popup-blocked' || error.code === 'auth/operation-not-supported-in-this-environment') return signInWithRedirect(auth, provider);
    throw error;
  }
}
export async function restoreSignedInAccount(){
  await authReady;
  await auth.authStateReady();
  return auth.currentUser;
}
export async function accountHeaders() {
  await authReady.catch(() => {});
  const token = await auth.currentUser?.getIdToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
export async function linkVerification() {
  const response = await fetch('/api/access', { method: 'POST', headers: { 'Content-Type': 'application/json', ...await accountHeaders() }, body: JSON.stringify({ action: 'link' }) });
  if (!response.ok) throw new Error('Account verification could not sync. Retry when online.');
}
