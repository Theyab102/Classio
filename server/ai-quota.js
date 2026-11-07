const { createHmac, randomBytes } = require("node:crypto");
const { isIP } = require("node:net");
const { cert, getApps, initializeApp } = require("firebase-admin/app");
const { getFirestore, Timestamp } = require("firebase-admin/firestore");

const WINDOW_MS = 10 * 60 * 1000, DAY_MS = 24 * 60 * 60 * 1000;
const instanceSecret = randomBytes(32);
const counters = new Map();

function quotaMode(env = process.env) {
  return env.AI_QUOTA_MODE || (env.FIREBASE_SERVICE_ACCOUNT_JSON ? "firestore" : "instance");
}
function configured(env = process.env) {
  if (quotaMode(env) === "instance") return true;
  if (quotaMode(env) !== "firestore" || (env.AI_RATE_LIMIT_SECRET || "").length < 32) return false;
  try {
    const account = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT_JSON || "");
    return ["project_id", "client_email", "private_key"].every(name => typeof account[name] === "string" && account[name].trim());
  } catch { return false; }
}
function requester(req, env = process.env) {
  const value = env.VERCEL && env.VERCEL_ENV !== "development" ? (req.headers["x-vercel-forwarded-for"] || req.headers["x-real-ip"] || req.headers["x-forwarded-for"]) : req.socket?.remoteAddress;
  const address = typeof value === "string" ? value.split(",")[0].trim() : "";
  if (!isIP(address)) return null;
  return createHmac("sha256", env.AI_RATE_LIMIT_SECRET || instanceSecret).update(address).digest("hex");
}
function specifications(action, id, now) {
  const audio = action === "audio";
  return [
    { key: `${action}-client-${id}`, interval: WINDOW_MS, limit: audio ? 3 : 18 },
    { key: `${action}-all`, interval: DAY_MS, limit: audio ? 40 : 250 },
  ].map(item => ({ ...item, start: Math.floor(now / item.interval) * item.interval, end: (Math.floor(now / item.interval) + 1) * item.interval }));
}
function reserveInstance(action, id, now = Date.now(), store = counters) {
  for (const [key, value] of store) if (value.end <= now) store.delete(key);
  const buckets = specifications(action, id, now);
  const full = buckets.filter(item => (store.get(item.key)?.count || 0) >= item.limit);
  if (full.length) return { retryAfter: Math.ceil((Math.max(...full.map(item => item.end)) - now) / 1000) };
  // Bound memory even under many distinct addresses. Fail closed when full.
  if (store.size + buckets.filter(item => !store.has(item.key)).length > 10000) return { unavailable: true };
  buckets.forEach(item => store.set(item.key, { count: (store.get(item.key)?.count || 0) + 1, end: item.end }));
  return {};
}
async function reserveRequest(req, action) {
  const id = requester(req);
  if (!id || !configured()) return { unavailable: true };
  if (quotaMode() === "instance") return reserveInstance(action, id);
  const name = "classio-api";
  const app = getApps().find(item => item.name === name) ?? initializeApp({ credential: cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON)) }, name);
  const collection = getFirestore(app).collection("aiRateLimits"), now = Date.now();
  const buckets = specifications(action, id, now).map(item => ({ ...item, ref: collection.doc(item.key) }));
  return getFirestore(app).runTransaction(async transaction => {
    const snapshots = await transaction.getAll(...buckets.map(item => item.ref));
    const counts = snapshots.map((snapshot, index) => snapshot.data()?.start === buckets[index].start ? snapshot.data().count : 0);
    const full = buckets.filter((item, index) => counts[index] >= item.limit);
    if (full.length) return { retryAfter: Math.ceil((Math.max(...full.map(item => item.end)) - now) / 1000) };
    buckets.forEach((item, index) => transaction.set(item.ref, { start: item.start, count: counts[index] + 1, expiresAt: Timestamp.fromMillis(item.end) }));
    return {};
  });
}
module.exports = { configured, reserveRequest, reserveInstance, quotaMode };
