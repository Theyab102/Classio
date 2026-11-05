const { createHmac } = require("node:crypto");
const { isIP } = require("node:net");
const { cert, getApps, initializeApp } = require("firebase-admin/app");
const { getFirestore, Timestamp } = require("firebase-admin/firestore");

const WINDOW_MS = 10 * 60 * 1000;
const DAY_MS = 24 * 60 * 60 * 1000;

function requester(req) {
  const value = process.env.VERCEL && process.env.VERCEL_ENV !== "development"
    ? req.headers["x-vercel-forwarded-for"]
    : req.socket?.remoteAddress;
  const address = typeof value === "string" ? value.split(",")[0].trim() : "";
  if (!isIP(address)) return null;
  return createHmac("sha256", process.env.AI_RATE_LIMIT_SECRET).update(address).digest("hex");
}

function configured() {
  return Boolean(process.env.GROQ_API_KEY && process.env.FIREBASE_SERVICE_ACCOUNT_JSON &&
    (process.env.AI_RATE_LIMIT_SECRET || "").length >= 32);
}

function bucket(key, now, interval, limit, collection) {
  const start = Math.floor(now / interval) * interval;
  return { ref: collection.doc(key), start, end: start + interval, limit };
}

async function reserveRequest(req, action) {
  const id = requester(req);
  if (!id) return { unavailable: true };
  const name = "classio-api";
  const app = getApps().find(item => item.name === name) ??
    initializeApp({ credential: cert(JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON)) }, name);
  const database = getFirestore(app);
  const collection = database.collection("aiRateLimits");
  const now = Date.now();
  const transcription = action === "audio";
  const buckets = [
    bucket(`${action}-client-${id}`, now, WINDOW_MS, transcription ? 3 : 18, collection),
    bucket(`${action}-all`, now, DAY_MS, transcription ? 40 : 250, collection),
  ];

  return database.runTransaction(async transaction => {
    const snapshots = await transaction.getAll(...buckets.map(item => item.ref));
    const counts = snapshots.map((snapshot, index) => {
      const previous = snapshot.data();
      return previous?.start === buckets[index].start ? previous.count : 0;
    });
    const exceeded = buckets.filter((item, index) => counts[index] >= item.limit);
    if (exceeded.length) return {
      retryAfter: Math.ceil((Math.max(...exceeded.map(item => item.end)) - now) / 1000),
    };
    buckets.forEach((item, index) => transaction.set(item.ref, {
      start: item.start, count: counts[index] + 1, expiresAt: Timestamp.fromMillis(item.end),
    }));
    return {};
  });
}

module.exports = { configured, reserveRequest };
