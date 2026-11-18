// Actual Firebase Admin JWT verification with ephemeral RSA test keys. Only
// Google's public certificate transport is replaced; no auth bypass/emulator.
const { test, after } = require("node:test"),
  assert = require("node:assert/strict"),
  crypto = require("node:crypto"),
  path = require("node:path");
const { initializeApp, deleteApp } = require("firebase-admin/app"),
  { getAuth } = require("firebase-admin/auth");
const { authorize } = require("./ai-auth");
const project = "classio-4378f",
  app = initializeApp({ projectId: project }, "classio-auth:" + project),
  auth = getAuth(app),
  keys = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
auth.idTokenVerifier.signatureVerifier.keyFetcher.fetchPublicKeys =
  async () => ({
    "fixture-public-key": keys.publicKey.export({
      type: "spki",
      format: "pem",
    }),
  });
after(() => deleteApp(app));
function token(patch = {}, key = keys.privateKey) {
  const now = Math.floor(Date.now() / 1000),
    header = Buffer.from(
      JSON.stringify({ alg: "RS256", kid: "fixture-public-key", typ: "JWT" }),
    ).toString("base64url"),
    payload = Buffer.from(
      JSON.stringify({
        iss: "https://securetoken.google.com/" + project,
        aud: project,
        sub: "fixture-student",
        iat: now - 10,
        exp: now + 3600,
        auth_time: now - 10,
        firebase: { sign_in_provider: "google.com" },
        ...patch,
      }),
    ).toString("base64url"),
    body = header + "." + payload;
  return (
    body +
    "." +
    crypto.sign("RSA-SHA256", Buffer.from(body), key).toString("base64url")
  );
}
const req = (value) => ({ headers: { authorization: "Bearer " + value } });
test("real signed Firebase ID tokens authorize with an exact project/provider identity", async () => {
  assert.deepEqual(await authorize(req(token()), { env: {} }), {
    sub: "user:fixture-student",
    kind: "account",
  });
});
test("expired, forged and wrong-project JWTs fail actual cryptographic verification", async () => {
  const other = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
  for (const value of [
    token({ exp: Math.floor(Date.now() / 1000) - 60 }),
    token({}, other.privateKey),
    token({ aud: "wrong-project" }),
  ])
    await assert.rejects(
      authorize(req(value), { env: {} }),
      (error) => error.status === 401,
    );
});
test("a Firebase certificate outage produces a service error, without instructing a valid user to sign out", async () => {
  const original =
    auth.idTokenVerifier.signatureVerifier.keyFetcher.fetchPublicKeys;
  auth.idTokenVerifier.signatureVerifier.keyFetcher.fetchPublicKeys =
    async () => {
      throw Error("Error fetching public keys for Google certs: offline");
    };
  try {
    await assert.rejects(
      authorize(req(token()), { env: {} }),
      (error) => error.status === 503 && error.code === "AUTH_UNAVAILABLE",
    );
  } finally {
    auth.idTokenVerifier.signatureVerifier.keyFetcher.fetchPublicKeys =
      original;
  }
});
test("a mismatched administrator project and a production emulator setting fail as configuration errors", async () => {
  for (const env of [
    {
      FIREBASE_SERVICE_ACCOUNT_JSON: JSON.stringify({
        project_id: "wrong-project",
      }),
    },
    { VERCEL: "1", FIREBASE_AUTH_EMULATOR_HOST: "127.0.0.1:9099" },
  ])
    await assert.rejects(
      authorize(req(token()), { env }),
      (error) => error.status === 503 && error.code === "AI_NOT_CONFIGURED",
    );
});
