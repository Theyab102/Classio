const { createHmac, randomBytes, timingSafeEqual } = require("node:crypto");
const COOKIE = "__Host-classio_ai";
class AccessError extends Error {
  constructor(status, code, reason) {
    super(code);
    this.status = status;
    this.code = code;
    this.reason = reason;
  }
}
function secret(env) {
  if ((env.CLASSIO_SESSION_SECRET || "").length < 32)
    throw new AccessError(503, "AI_NOT_CONFIGURED");
  return env.CLASSIO_SESSION_SECRET;
}
function signature(value, env) {
  return createHmac("sha256", secret(env)).update(value).digest("base64url");
}
function equal(a, b) {
  if (typeof a !== "string" || typeof b !== "string") return false;
  const left = Buffer.from(a),
    right = Buffer.from(b);
  return left.length === right.length && timingSafeEqual(left, right);
}
function cookie(req) {
  return (req.headers.cookie || "")
    .split(";")
    .map((item) => item.trim())
    .find((item) => item.startsWith(COOKIE + "="))
    ?.slice(COOKIE.length + 1);
}
function readSession(req, env = process.env, now = Date.now()) {
  const token = cookie(req);
  if (!token || token.length > 1200 || !/^[-\w]+\.[-\w]+$/.test(token))
    return null;
  const [payload, mac] = token.split(".");
  if (!mac || !equal(mac, signature(payload, env))) return null;
  try {
    const value = JSON.parse(
      Buffer.from(payload, "base64url").toString("utf8"),
    );
    if (
      value.exp <= now ||
      value.exp > now + 13 * 60 * 60 * 1000 ||
      !/^guest_[a-zA-Z0-9_-]{24,64}$/.test(value.sub) ||
      typeof value.nonce !== "string"
    )
      return null;
    return value;
  } catch {
    return null;
  }
}
function sameOrigin(req) {
  const site = req.headers["sec-fetch-site"];
  if (site && site !== "same-origin")
    throw new AccessError(403, "ORIGIN_REJECTED");
  const origin = req.headers.origin;
  if (origin) {
    let url;
    try {
      url = new URL(origin);
    } catch {
      throw new AccessError(403, "ORIGIN_REJECTED");
    }
    if (url.host !== (req.headers.host || req.headers["x-forwarded-host"]))
      throw new AccessError(403, "ORIGIN_REJECTED");
  }
}
function createGuestSession(req, res, env = process.env) {
  sameOrigin(req);
  const existing = readSession(req, env),
    value = existing || {
      sub: "guest_" + randomBytes(24).toString("base64url"),
      nonce: randomBytes(24).toString("base64url"),
      exp: Date.now() + 12 * 60 * 60 * 1000,
    };
  const payload = Buffer.from(JSON.stringify(value)).toString("base64url"),
    token = payload + "." + signature(payload, env);
  res.setHeader(
    "Set-Cookie",
    `${COOKIE}=${token}; Path=/; HttpOnly; Secure; SameSite=Strict`,
  );
  return { csrf: signature("csrf:" + value.nonce, env), expiresAt: value.exp };
}
async function verifyFirebase(token, env) {
  if (
    (env.VERCEL || env.NODE_ENV === "production") &&
    env.FIREBASE_AUTH_EMULATOR_HOST
  )
    throw new AccessError(503, "AI_NOT_CONFIGURED");
  const { getApps, initializeApp, cert } = require("firebase-admin/app"),
    { getAuth } = require("firebase-admin/auth");
  let credential;
  try {
    const account = env.FIREBASE_SERVICE_ACCOUNT_JSON
      ? JSON.parse(env.FIREBASE_SERVICE_ACCOUNT_JSON)
      : null;
    if (
      account &&
      (account.project_id || account.projectId) !==
        (env.FIREBASE_PROJECT_ID || "classio-4378f")
    )
      throw new AccessError(503, "AI_NOT_CONFIGURED", "ADMIN_PROJECT_MISMATCH");
    credential = account ? cert(account) : undefined;
  } catch (error) {
    if (error instanceof AccessError) throw error;
    throw new AccessError(503, "AI_NOT_CONFIGURED", "ADMIN_CREDENTIAL_INVALID");
  }
  const projectId = env.FIREBASE_PROJECT_ID || "classio-4378f",
    name = "classio-auth:" + projectId;
  const app =
    getApps().find((app) => app.name === name) ||
    initializeApp({ projectId, ...(credential ? { credential } : {}) }, name);
  return getAuth(app).verifyIdToken(token, !!credential);
}
function verificationError(error) {
  if (error instanceof AccessError) return error;
  if (
    /^(?:Error fetching public keys|Error while making request)/.test(
      error?.message || "",
    )
  )
    return new AccessError(
      503,
      "AUTH_UNAVAILABLE",
      "FIREBASE_CERTIFICATE_LOOKUP_FAILED",
    );
  if (error?.code === "auth/id-token-expired")
    return new AccessError(401, "AUTH_EXPIRED");
  if (
    [
      "auth/id-token-revoked",
      "auth/user-disabled",
      "auth/user-not-found",
    ].includes(error?.code)
  )
    return new AccessError(401, "AUTH_REVOKED");
  if (
    [
      "auth/argument-error",
      "auth/invalid-argument",
      "auth/invalid-id-token",
    ].includes(error?.code)
  )
    return new AccessError(401, "AUTH_REQUIRED");
  if (
    [
      "app/invalid-credential",
      "auth/invalid-credential",
      "auth/insufficient-permission",
      "app/invalid-app-options",
    ].includes(error?.code)
  )
    return new AccessError(503, "AI_NOT_CONFIGURED");
  // Certificate lookup, administrator permission/network errors and outages
  // cannot be fixed by signing the student out. Never mislabel them as 401.
  return new AccessError(503, "AUTH_UNAVAILABLE");
}
async function authorize(
  req,
  { env = process.env, verify = verifyFirebase } = {},
) {
  sameOrigin(req);
  const bearer = /^Bearer ([^\s]{20,10000})$/.exec(
    req.headers.authorization || "",
  );
  if (req.headers.authorization && !bearer)
    throw new AccessError(401, "AUTH_REQUIRED");
  if (bearer) {
    let user, timer;
    try {
      user = await Promise.race([
        verify(bearer[1], env),
        new Promise((_, reject) => {
          timer = setTimeout(
            () => reject(new AccessError(503, "AUTH_UNAVAILABLE")),
            8000,
          );
        }),
      ]);
    } catch (error) {
      throw verificationError(error);
    } finally {
      clearTimeout(timer);
    }
    if (!user.uid || user.firebase?.sign_in_provider !== "google.com")
      throw new AccessError(403, "AUTH_PROVIDER_REQUIRED");
    return { sub: "user:" + user.uid, kind: "account" };
  }
  const session = readSession(req, env);
  if (
    !session ||
    !equal(
      req.headers["x-classio-csrf"],
      signature("csrf:" + session.nonce, env),
    )
  )
    throw new AccessError(401, "GUEST_SESSION_REQUIRED");
  return { sub: session.sub, kind: "guest" };
}
module.exports = {
  AccessError,
  authorize,
  createGuestSession,
  readSession,
  sameOrigin,
  verificationError,
};
