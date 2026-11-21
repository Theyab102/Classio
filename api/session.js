const {
  createGuestSession,
  readSession,
  AccessError,
} = require("../server/ai-auth");
const { reserveRequest } = require("../server/ai-quota");
const { accessFailure } = require("../server/access-errors");
const { accessDiagnostic } = require("../server/access-diagnostic");
module.exports = async function session(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST." });
  }
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers["content-type"] || ""))
    return res.status(415).json({ error: "Send JSON." });
  try {
    require("../server/ai-auth").sameOrigin(req);
    // Refreshing a valid session keeps its identity and must not consume the
    // limited allowance for issuing new anonymous identities.
    if (readSession(req))
      return res.status(200).json(createGuestSession(req, res));
    const quota = await reserveRequest(req, "session");
    if (quota.unavailable)
      return res.status(503).json({
        code: "AI_NOT_CONFIGURED",
        error: "The study service is not configured yet. Your writing is kept.",
      });
    if (quota.retryAfter) {
      res.setHeader("Retry-After", String(quota.retryAfter));
      return res.status(429).json({
        code: "CLIENT_RATE_LIMIT",
        error: "Please wait before starting another AI session.",
      });
    }
    return res.status(200).json(createGuestSession(req, res));
  } catch (error) {
    if (error instanceof AccessError) {
      const failure = accessFailure(error);
      console.warn(
        "session_access",
        accessDiagnostic(error, failure.body.code),
      );
      return res.status(failure.status).json(failure.body);
    }
    return res.status(503).json({
      code: "SESSION_UNAVAILABLE",
      error:
        "A secure study session could not be started. Your writing is kept.",
    });
  }
};
