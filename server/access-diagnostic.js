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
const references = {
  AUTH_EMULATOR_REJECTED: "A300",
  ADMIN_PROJECT_MISMATCH: "A302",
  ADMIN_CREDENTIAL_INVALID: "A303",
  FIREBASE_CERTIFICATE_LOOKUP_FAILED: "A201",
  FIREBASE_ADMIN_CONFIGURATION_FAILED: "A301",
  FIREBASE_SDK_LOAD_FAILED: "A100",
  FIREBASE_NETWORK_FAILED: "A202",
  FIREBASE_INTERNAL_FAILED: "A401",
  FIREBASE_VERIFICATION_FAILED: "A500",
  FIREBASE_VERIFICATION_TIMEOUT: "A408",
};
const allowedReferences = new Set([
  ...Object.values(references),
  "A101",
  "A102",
  "A103",
  "A104",
]);
function supportReference(error) {
  return allowedReferences.has(error?.supportReference)
    ? error.supportReference
    : Object.hasOwn(references, error?.reason)
      ? references[error.reason]
      : undefined;
}
function accessDiagnostic(error, code) {
  const reference = supportReference(error);
  return {
    code,
    ...(reasons.has(error?.reason) ? { reason: error.reason } : {}),
    ...(phases.has(error?.phase) ? { phase: error.phase } : {}),
    ...(reference ? { reference } : {}),
  };
}
module.exports = { accessDiagnostic, supportReference };
