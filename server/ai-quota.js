const { createHmac, randomBytes } = require("node:crypto");
const { isIP } = require("node:net");
const { cert, getApps, initializeApp } = require("firebase-admin/app");
const { getFirestore, Timestamp } = require("firebase-admin/firestore");
const counters = new Map(),
  leases = new Map(),
  instanceSecret = randomBytes(32),
  WINDOW = 600000,
  DAY = 86400000;
function quotaMode(env = process.env) {
  return (
    env.AI_QUOTA_MODE ||
    (env.VERCEL && env.VERCEL_ENV !== "development"
      ? "firestore"
      : env.FIREBASE_SERVICE_ACCOUNT_JSON
        ? "firestore"
        : "instance")
  );
}
function configured(env = process.env) {
  if (quotaMode(env) === "instance")
    return !env.VERCEL || env.VERCEL_ENV === "development";
  if (
    quotaMode(env) !== "firestore" ||
    (env.AI_RATE_LIMIT_SECRET || env.CLASSIO_SESSION_SECRET || "").length < 32
  )
    return false;
  try {
    const account = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT_JSON || "");
    return ["project_id", "client_email", "private_key"].every(
      (name) => typeof account[name] === "string" && account[name].trim(),
    );
  } catch {
    return false;
  }
}
function ids(req, env = process.env) {
  const raw =
      env.VERCEL && env.VERCEL_ENV !== "development"
        ? req.headers["x-vercel-forwarded-for"] ||
          req.headers["x-real-ip"] ||
          req.headers["x-forwarded-for"]
        : req.socket?.remoteAddress,
    address = typeof raw === "string" ? raw.split(",")[0].trim() : "";
  if (!isIP(address)) return null;
  const key =
      env.AI_RATE_LIMIT_SECRET || env.CLASSIO_SESSION_SECRET || instanceSecret,
    hmac = (text) => createHmac("sha256", key).update(text).digest("hex");
  return {
    ip: hmac("ip:" + address),
    user: req.aiIdentity ? hmac("user:" + req.aiIdentity.sub) : null,
  };
}
function specs(action, identity, now, env = {}) {
  const session = action === "session",
    audio = action === "audio",
    image = action === "image";
  const limit = session
    ? 8
    : audio
      ? 3
      : image
        ? 3
        : Math.min(
            100,
            Math.max(1, Number(env.AI_USER_REQUESTS_PER_10_MIN) || 40),
          );
  return [
    {
      key: action + "-ip-" + identity.ip,
      interval: session ? 3600000 : WINDOW,
      limit,
    },
    ...(identity.user
      ? [{ key: action + "-user-" + identity.user, interval: WINDOW, limit }]
      : []),
    {
      key: action + "-all",
      interval: DAY,
      limit: session
        ? 1000
        : audio
          ? 40
          : image
            ? 30
            : Math.min(
                1000,
                Math.max(1, Number(env.AI_APP_REQUESTS_PER_DAY) || 250),
              ),
    },
  ].map((row) => ({
    ...row,
    start: Math.floor(now / row.interval) * row.interval,
    end: (Math.floor(now / row.interval) + 1) * row.interval,
  }));
}
function reserveInstance(
  action,
  id,
  now = Date.now(),
  store = counters,
  env = {},
) {
  const identity = typeof id === "string" ? { ip: id } : id;
  for (const [key, value] of store) if (value.end <= now) store.delete(key);
  const buckets = specs(action, identity, now, env),
    full = buckets.filter(
      (row) => (store.get(row.key)?.count || 0) >= row.limit,
    );
  if (full.length)
    return {
      retryAfter: Math.ceil(
        (Math.max(...full.map((row) => row.end)) - now) / 1000,
      ),
    };
  if (store.size + buckets.filter((row) => !store.has(row.key)).length > 10000)
    return { unavailable: true };
  buckets.forEach((row) =>
    store.set(row.key, {
      count: (store.get(row.key)?.count || 0) + 1,
      end: row.end,
    }),
  );
  return {};
}
function reserveConcurrent(identity, now) {
  for (const [key, value] of leases) if (value.until < now) leases.delete(key);
  const user = identity.user || identity.ip;
  if (
    [...leases.values()].filter((row) => row.user === user).length >= 2 ||
    leases.size >= 8
  )
    return null;
  const token = randomBytes(16).toString("hex");
  leases.set(token, { user, until: now + 60000 });
  return () => leases.delete(token);
}
async function reserveRequest(req, action) {
  const inference = ["study", "audio", "image"].includes(action),
    identity = ids(req);
  if (!identity || !configured()) return { unavailable: true };
  if (quotaMode() === "instance") {
    const quota = reserveInstance(
      action,
      identity,
      Date.now(),
      counters,
      process.env,
    );
    if (quota.unavailable || quota.retryAfter) return quota;
    if (!inference) return {};
    const release = reserveConcurrent(identity, Date.now());
    return release ? { release } : { retryAfter: 15 };
  }
  const app =
      getApps().find((app) => app.name === "classio-api") ||
      initializeApp(
        {
          credential: cert(
            JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON),
          ),
        },
        "classio-api",
      ),
    db = getFirestore(app),
    collection = db.collection("aiRateLimits"),
    now = Date.now(),
    token = randomBytes(16).toString("hex");
  const buckets = specs(action, identity, now, process.env).map((row) => ({
    ...row,
    ref: collection.doc(row.key),
  }));
  const appRef = collection.doc("concurrent-all"),
    userRef = collection.doc("concurrent-" + (identity.user || identity.ip));
  const result = await db.runTransaction(async (transaction) => {
    const snapshots = await transaction.getAll(
        ...buckets.map((row) => row.ref),
        appRef,
        userRef,
      ),
      counts = snapshots
        .slice(0, buckets.length)
        .map((snapshot, index) =>
          snapshot.data()?.start === buckets[index].start
            ? snapshot.data().count
            : 0,
        ),
      full = buckets.filter((row, index) => counts[index] >= row.limit);
    if (full.length)
      return {
        retryAfter: Math.ceil(
          (Math.max(...full.map((row) => row.end)) - now) / 1000,
        ),
      };
    if (!inference) {
      buckets.forEach((row, index) =>
        transaction.set(row.ref, {
          start: row.start,
          count: counts[index] + 1,
          expiresAt: Timestamp.fromMillis(row.end),
        }),
      );
      return { counterOnly: true };
    }
    const active = snapshots
      .slice(buckets.length)
      .map((snapshot) =>
        (snapshot.data()?.leases || []).filter((row) => row.until > now),
      );
    if (active[0].length >= 8 || active[1].length >= 2)
      return { retryAfter: 15 };
    buckets.forEach((row, index) =>
      transaction.set(row.ref, {
        start: row.start,
        count: counts[index] + 1,
        expiresAt: Timestamp.fromMillis(row.end),
      }),
    );
    [appRef, userRef].forEach((ref, index) =>
      transaction.set(ref, {
        leases: [...active[index], { token, until: now + 60000 }],
        expiresAt: Timestamp.fromMillis(now + 120000),
      }),
    );
    return {};
  });
  if (result.retryAfter) return result;
  if (result.counterOnly) return {};
  return {
    release: async () => {
      try {
        await db.runTransaction(async (transaction) => {
          const snapshots = await transaction.getAll(appRef, userRef);
          snapshots.forEach((snapshot, index) =>
            transaction.set([appRef, userRef][index], {
              leases: (snapshot.data()?.leases || []).filter(
                (row) => row.token !== token && row.until > Date.now(),
              ),
              expiresAt: Timestamp.fromMillis(Date.now() + 120000),
            }),
          );
        });
      } catch {
        /* Expiring leases still bound concurrency if release is interrupted. */
      }
    },
  };
}
module.exports = { quotaMode, configured, reserveRequest, reserveInstance };
