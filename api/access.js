const { jsonRequest, reject } = require('../server/http');
const { readAccess, setAccess, authenticatedUser, TERMS_VERSION } = require('../server/access');
const admin = require('../server/admin');

function ageDecision(value, today = new Date()) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value || '');
  if (!match) return null;
  const [, y, m, d] = match.map(Number);
  const date = new Date(y, m - 1, d);
  if (y < 1900 || date > today || date.getFullYear() !== y || date.getMonth() !== m - 1 || date.getDate() !== d) return null;
  const age = today.getFullYear() - y - (today.getMonth() + 1 < m || (today.getMonth() + 1 === m && today.getDate() < d) ? 1 : 0);
  return { stage: age >= 13 ? 'eligible' : 'blocked', adult: age >= 18 };
}

module.exports = async (req, res) => {
  if (!jsonRequest(req, res, ['GET', 'POST'])) return;
  if (req.method === 'GET') return res.status(200).json({ stage: readAccess(req)?.stage || 'new' });
  try {
    const previous = readAccess(req);
    const action = req.body?.action;
    let claim;
    if (action === 'screen') {
      if (previous?.stage === 'blocked') return reject(res, 403, 'AGE_RESTRICTED', 'Classio is unavailable to you.');
      claim = ageDecision(req.body.birthDate);
      if (!claim) return reject(res, 400, 'INVALID_DATE', 'Enter a valid date of birth.');
    } else if (action === 'accept') {
      if (previous?.stage !== 'eligible' || req.body.termsAccepted !== true) return reject(res, 400, 'TERMS_REQUIRED', 'Complete eligibility and accept the terms.');
      claim = { stage: 'accepted', adult: previous.adult === true };
    } else if (action === 'restore' || action === 'link') {
      const user = await authenticatedUser(req, true);
      const ref = admin.database().doc(`users/${user.uid}/access/verification`);
      const snap = await ref.get();
      const stored = snap.data();
      if (stored?.stage === 'accepted' && stored.version === TERMS_VERSION) {
        claim = { stage: 'accepted', adult: stored.adult === true };
      } else if (action === 'link' && previous?.stage === 'accepted') {
        claim = { stage: 'accepted', adult: previous.adult === true };
        await ref.set({ ...claim, version: TERMS_VERSION, verifiedAt: new Date().toISOString(), method: 'self-declared' });
      } else {
        return res.status(200).json({ stage: 'new' });
      }
    } else return reject(res, 400, 'INVALID_ACTION', 'This verification request is invalid.');
    setAccess(res, claim);
    return res.status(200).json({ stage: claim.stage });
  } catch (error) {
    if (error.status) return reject(res, error.status, error.code, error.message);
    console.error('Classio access configuration unavailable', { code: error.message === 'SESSION_NOT_CONFIGURED' ? 'SESSION_NOT_CONFIGURED' : 'ACCOUNT_VERIFICATION_UNAVAILABLE' });
    return reject(res, 503, 'VERIFICATION_UNAVAILABLE', 'Verification is temporarily unavailable. Your progress has been kept. Try again.');
  }
};
module.exports.ageDecision = ageDecision;
