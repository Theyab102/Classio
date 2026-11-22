// Only fixed, public messages leave an authorization boundary.
const { supportReference } = require("./access-diagnostic");
const messages = {
  AUTH_EXPIRED:
    "Your sign-in session expired. Retry to refresh it. Your work is kept.",
  AUTH_REQUIRED:
    "Classio could not verify this sign-in. Sign in again if retrying does not help. Your work is kept.",
  AUTH_REVOKED:
    "This sign-in is no longer valid. Sign in again to continue. Your work is kept.",
  AUTH_PROVIDER_REQUIRED:
    "Sign in with your Google account to use AI. Your work is kept.",
  GUEST_SESSION_REQUIRED:
    "Your guest AI session could not be restored. Allow cookies for Classio, then retry. Your work is kept.",
  AUTH_UNAVAILABLE:
    "Classio’s sign-in check is temporarily unavailable. Try again shortly. Your work is kept.",
  AI_NOT_CONFIGURED:
    "AI tools are waiting for the site’s server setup. Your work is kept.",
  ORIGIN_REJECTED:
    "Open Classio at its normal website address, then retry. Your work is kept.",
};
function accessFailure(error) {
  const code = Object.hasOwn(messages, error?.code)
    ? error.code
    : "AUTH_UNAVAILABLE";
  const reference =
    code === "AUTH_UNAVAILABLE" || code === "AI_NOT_CONFIGURED"
      ? supportReference(error) ||
        (code === "AI_NOT_CONFIGURED" ? "A300" : "A500")
      : null;
  return {
    status:
      code === "AI_NOT_CONFIGURED" || code === "AUTH_UNAVAILABLE"
        ? 503
        : code === "ORIGIN_REJECTED" || code === "AUTH_PROVIDER_REQUIRED"
          ? 403
          : 401,
    body: {
      code,
      error: messages[code] + (reference ? ` Reference: ${reference}.` : ""),
    },
  };
}
module.exports = { accessFailure };
