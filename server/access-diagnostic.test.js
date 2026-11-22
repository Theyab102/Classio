const { test } = require("node:test"),
  assert = require("node:assert/strict");
const { verificationError, authorize } = require("./ai-auth");
const { accessDiagnostic } = require("./access-diagnostic");
const { accessFailure } = require("./access-errors");

test("SDK, network and administrator failures retain fixed private diagnostics", () => {
  for (const [code, reason, reference] of [
    ["MODULE_NOT_FOUND", "FIREBASE_SDK_LOAD_FAILED", "A101"],
    ["ERR_REQUIRE_ASYNC_MODULE", "FIREBASE_SDK_LOAD_FAILED", "A103"],
    ["ERR_PACKAGE_PATH_NOT_EXPORTED", "FIREBASE_SDK_LOAD_FAILED", "A104"],
    ["app/network-error", "FIREBASE_NETWORK_FAILED", "A202"],
    [
      "auth/insufficient-permission",
      "FIREBASE_ADMIN_CONFIGURATION_FAILED",
      "A301",
    ],
    ["auth/internal-error", "FIREBASE_INTERNAL_FAILED", "A401"],
  ]) {
    const error = verificationError(
      Object.assign(Error("private token, path and provider response"), {
        code,
      }),
      "sdk_load",
    );
    assert.equal(error.reason, reason);
    assert.deepEqual(accessDiagnostic(error, error.code), {
      code: error.code,
      reason,
      phase: "sdk_load",
      reference,
    });
    assert(!JSON.stringify(error).includes("private"));
    assert.deepEqual(Object.keys(accessFailure(error).body).sort(), [
      "code",
      "error",
    ]);
    assert(
      accessFailure(error).body.error.endsWith(`Reference: ${reference}.`),
    );
  }
});
test("unknown diagnostic properties and raw errors cannot reach logs or public responses", () => {
  const raw = Object.assign(Error("private token and internal path"), {
    reason: "private token",
    phase: "private path",
  });
  const safe = verificationError(raw);
  assert.equal(safe.reason, "FIREBASE_VERIFICATION_FAILED");
  assert.deepEqual(accessDiagnostic(raw, "AUTH_UNAVAILABLE"), {
    code: "AUTH_UNAVAILABLE",
  });
  assert(
    !JSON.stringify(accessDiagnostic(safe, safe.code)).includes("private"),
  );
  assert(!JSON.stringify(accessFailure(safe)).includes("private"));
});
test("production credential parsing and project mismatch report the credential phase without disclosing JSON", async () => {
  for (const [account, reason, reference] of [
    ["private invalid JSON", "ADMIN_CREDENTIAL_INVALID", "A303"],
    [
      JSON.stringify({
        project_id: "wrong-project",
        private_key: "private-key",
      }),
      "ADMIN_PROJECT_MISMATCH",
      "A302",
    ],
  ]) {
    await assert.rejects(
      authorize(
        {
          headers: { authorization: "Bearer fixture-token-at-least-20-chars" },
        },
        { env: { FIREBASE_SERVICE_ACCOUNT_JSON: account } },
      ),
      (error) => {
        assert.deepEqual(accessDiagnostic(error, error.code), {
          code: "AI_NOT_CONFIGURED",
          reason,
          phase: "credential_configuration",
          reference,
        });
        assert(!JSON.stringify(accessFailure(error)).includes("private"));
        return true;
      },
    );
  }
});
test("support references are allowlisted and never publish arbitrary diagnostic properties", () => {
  for (const reason of [
    "__proto__",
    "constructor",
    "toString",
    "private credential",
  ]) {
    const body = accessFailure({
      code: "AUTH_UNAVAILABLE",
      reason,
      supportReference: "private token",
    }).body;
    assert(body.error.endsWith("Reference: A500."));
    assert(!body.error.includes("private"));
    assert(!body.error.includes("function"));
  }
});
test("expired and revoked credentials stay unauthorized rather than becoming guest identities", async () => {
  for (const [code, expected] of [
    ["auth/id-token-expired", "AUTH_EXPIRED"],
    ["auth/id-token-revoked", "AUTH_REVOKED"],
  ])
    await assert.rejects(
      authorize(
        {
          headers: { authorization: "Bearer fixture-token-at-least-20-chars" },
        },
        {
          verify: async () => {
            throw Object.assign(Error("private"), { code });
          },
        },
      ),
      (error) => error.status === 401 && error.code === expected,
    );
});
