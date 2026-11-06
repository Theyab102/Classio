const { cert, getApps, initializeApp } = require('firebase-admin/app');
const { getFirestore } = require('firebase-admin/firestore');
const { getAuth } = require('firebase-admin/auth');

function adminApp() {
  const existing = getApps().find(app => app.name === 'classio-api');
  if (existing) return existing;
  if (process.env.FIREBASE_AUTH_EMULATOR_HOST && process.env.NODE_ENV !== 'production') {
    return initializeApp({ projectId: process.env.GCLOUD_PROJECT || 'demo-classio' }, 'classio-api');
  }
  if (!process.env.FIREBASE_SERVICE_ACCOUNT_JSON) throw new Error('FIREBASE_ADMIN_NOT_CONFIGURED');
  return initializeApp({ credential: cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON)) }, 'classio-api');
}

const database = () => getFirestore(adminApp());
const identity = () => getAuth(adminApp());
module.exports = { database, identity };
