function reject(res, status, code, message) {
  return res.status(status).json({ code, error: message });
}
function publicError(error) {
  return typeof error?.code === 'string' && new Set([
    'SIGN_IN_REQUIRED', 'INVALID_SESSION', 'VERIFICATION_REQUIRED', 'INVALID_PROVIDER',
    'PROVIDER_NOT_CONFIGURED', 'PROVIDER_AGE_RESTRICTED', 'MODEL_NOT_CONFIGURED', 'VISION_NOT_CONFIGURED',
    'PROVIDER_TIMEOUT', 'PROVIDER_RATE_LIMIT', 'PROVIDER_AUTH_FAILED',
    'PROVIDER_MODEL_UNAVAILABLE', 'PROVIDER_UNAVAILABLE', 'EMPTY_RESPONSE',
    'QUOTA_UNAVAILABLE', 'USAGE_LIMIT', 'INVALID_COURSE', 'FILE_CHANGED',
  ]).has(error.code);
}

function jsonRequest(req, res, methods = ['POST']) {
  res.setHeader('Cache-Control', 'no-store');
  if (!methods.includes(req.method)) {
    res.setHeader('Allow', methods.join(', '));
    reject(res, 405, 'METHOD_NOT_ALLOWED', 'This method is not supported.');
    return false;
  }
  const site = req.headers['sec-fetch-site'];
  if (site && site !== 'same-origin' && site !== 'none') {
    reject(res, 403, 'CROSS_SITE_REQUEST', 'Open Classio to use this feature.');
    return false;
  }
  const origin = req.headers.origin;
  if (origin) {
    let originHost;
    try { originHost = new URL(origin).host; } catch {}
    if (originHost !== req.headers.host && originHost !== req.headers['x-forwarded-host']) {
      reject(res, 403, 'CROSS_SITE_REQUEST', 'Open Classio to use this feature.');
      return false;
    }
  }
  if (req.method !== 'GET' && !/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] || '')) {
    reject(res, 415, 'JSON_REQUIRED', 'Send a JSON request.');
    return false;
  }
  return true;
}
module.exports = { jsonRequest, reject, publicError };
