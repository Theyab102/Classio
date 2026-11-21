// Allowlisted classifications only. Never log SDK errors, tokens or messages.
const reasons = new Set([
  "AUTH_EMULATOR_REJECTED",
  "ADMIN_PROJECT_MISMATCH",
  "ADMIN_CREDENTIAL_INVALID",
  "FIREBASE_CERTIFICATE_LOOKUP_FAILED",
  "FIREBASE_TOKEN_EXPIRED",
  "FIREBASE_ACCOUNT_REJECTED",
  "FIREBASE_TOKEN_INVALID",
  "FIREBASE_ADMIN_CONFIGURATION_FAILED",
  "FIREBASE_SDK_LOAD_FAILED",
  "FIREBASE_NETWORK_FAILED",
  "FIREBASE_INTERNAL_FAILED",
  "FIREBASE_VERIFICATION_FAILED",
  "FIREBASE_VERIFICATION_TIMEOUT",
]);
const phases = new Set([
  "configuration",
  "sdk_load",
  "credential_configuration",
  "app_initialization",
  "token_verification",
  "token_and_revocation_verification",
  "certificate_lookup",
  "verification_deadline",
]);
function accessDiagnostic(error, code) {
  return {
    code,
    ...(reasons.has(error?.reason) ? { reason: error.reason } : {}),
    ...(phases.has(error?.phase) ? { phase: error.phase } : {}),
  };
}
module.exports = { accessDiagnostic };
