const { createHmac, timingSafeEqual } = require('node:crypto');
const admin = require('./admin');

const COOKIE = 'classio_access_v2';
const TERMS_VERSION = '2026-11-06';
const YEAR_SECONDS = 365 * 24 * 60 * 60;

function signingKey() {
  const key = process.env.CLASSIO_SESSION_SECRET || process.env.AI_RATE_LIMIT_SECRET || '';
  if (key.length < 32) throw new Error('SESSION_NOT_CONFIGURED');
  return key;
}

function readAccess(req) {
  try {
    const raw = (req.headers.cookie || '').split(';').map(part => part.trim()).find(part => part.startsWith(COOKIE + '='))?.slice(COOKIE.length + 1);
    if (!raw || raw.length > 2000) return null;
    const [body, signature, extra] = raw.split('.');
    if (!body || !signature || extra) return null;
    const expected = createHmac('sha256', signingKey()).update(body).digest();
    const actual = Buffer.from(signature, 'base64url');
    if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) return null;
    const claim = JSON.parse(Buffer.from(body, 'base64url').toString());
    if (claim.exp <= Date.now() || claim.version !== TERMS_VERSION || !['eligible', 'accepted', 'blocked'].includes(claim.stage)) return null;
    return claim;
  } catch { return null; }
}

function setAccess(res, claim) {
  const body = Buffer.from(JSON.stringify({ ...claim, version: TERMS_VERSION, exp: Date.now() + YEAR_SECONDS * 1000 })).toString('base64url');
  const signature = createHmac('sha256', signingKey()).update(body).digest('base64url');
  const secure = process.env.NODE_ENV === 'production' || Boolean(process.env.VERCEL);
  res.setHeader('Set-Cookie', `${COOKIE}=${body}.${signature}; Max-Age=${YEAR_SECONDS}; Path=/; HttpOnly; SameSite=Lax${secure ? '; Secure' : ''}`);
}

async function authenticatedUser(req, required = false) {
  const header = req.headers.authorization;
  if (!header) {
    if (required) throw Object.assign(new Error('Sign in with Google to save to your account.'), { status: 401, code: 'SIGN_IN_REQUIRED' });
    return null;
  }
  try {
    if (typeof header !== 'string' || !header.startsWith('Bearer ') || header.length > 10000) throw new Error();
    const token = await admin.identity().verifyIdToken(header.slice(7), true);
    if (!token.uid || token.firebase?.sign_in_provider !== 'google.com' || !token.email_verified) throw new Error();
    return token;
  } catch {
    throw Object.assign(new Error('Your sign-in could not be verified. Sign in again.'), { status: 401, code: 'INVALID_SESSION' });
  }
}

async function requireAccess(req, { account = false } = {}) {
  const user = await authenticatedUser(req, account);
  if (user) {
    const snap = await admin.database().doc(`users/${user.uid}/access/verification`).get();
    const record = snap.data();
    if (record?.stage === 'accepted' && record.version === TERMS_VERSION) return { user, ...record };
  }
  const access = readAccess(req);
  if (access?.stage === 'accepted') return { user, ...access };
  throw Object.assign(new Error('Complete the eligibility and terms steps before using AI.'), { status: 403, code: 'VERIFICATION_REQUIRED' });
}

module.exports = { readAccess, setAccess, requireAccess, authenticatedUser, TERMS_VERSION };
