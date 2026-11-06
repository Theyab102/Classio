const { configured, reserveRequest } = require('../server/ai-quota');
const { parseAudioRequest, parseStudyRequest } = require('../server/study-request');
const { generate, transcribe, providers } = require('../server/ai-providers');
const { requireAccess } = require('../server/access');
const { jsonRequest, reject, publicError } = require('../server/http');

module.exports = async function study(req, res) {
  if (!jsonRequest(req, res, ['GET', 'POST'])) return;
  if (req.method === 'GET') return res.status(200).json({ providers: providers(), fallbackRequiresConsent: true });
  const audio = req.body?.kind === 'audio';
  const input = audio ? parseAudioRequest(req.body) : parseStudyRequest(req.body);
  if (!input || (req.body.provider !== undefined && !['auto', 'groq', 'gemini'].includes(req.body.provider)) ||
      (req.body.allowFallback !== undefined && typeof req.body.allowFallback !== 'boolean')) {
    return reject(res, 400, 'INVALID_REQUEST', 'This request is invalid or too large. Check the file or shorten the input.');
  }
  if (audio && req.body.provider === 'gemini') return reject(res, 400, 'AUDIO_PROVIDER_UNSUPPORTED', 'Audio transcription uses Groq. Choose Groq or Auto to permit transcription; your current Gemini preference has been respected.');
  try {
    if (!configured()) {
      const missing = [!process.env.FIREBASE_SERVICE_ACCOUNT_JSON && 'FIREBASE_SERVICE_ACCOUNT_JSON',
        (process.env.AI_RATE_LIMIT_SECRET || '').length < 32 && 'AI_RATE_LIMIT_SECRET'].filter(Boolean);
      console.error('Classio AI server configuration missing', { missing });
      return reject(res, 503, 'AI_CONFIG_MISSING', 'AI is not configured on this deployment. The operator must configure Firebase Admin, a rate-limit secret, and an AI provider, then redeploy.');
    }
    const access = await requireAccess(req);
    const quota = await reserveRequest(req, audio ? 'audio' : 'study', access.user?.uid);
    if (quota.unavailable) return reject(res, 503, 'QUOTA_UNAVAILABLE', 'AI usage limits could not be checked. Please retry later.');
    if (quota.retryAfter) {
      res.setHeader('Retry-After', String(quota.retryAfter));
      return reject(res, 429, 'USAGE_LIMIT', 'The free AI request limit was reached. Try again later.');
    }
    const result = audio ? await transcribe(input) : await generate(input, req.body);
    return res.status(200).json(result);
  } catch (error) {
    if (publicError(error)) return reject(res, error.status, error.code, error.message);
    console.error('Classio AI request failed', { code: 'AI_REQUEST_FAILED' });
    return reject(res, 503, 'AI_REQUEST_FAILED', 'AI could not complete this request. Your input has been kept; please retry.');
  }
};
