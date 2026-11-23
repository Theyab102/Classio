import { guestScope } from "./guestWorkspace";
let identity = () => null,
  guest = null,
  ready = () => undefined;
const refreshing = new Map();
export function setAiIdentityProvider(read, wait = () => undefined) {
  identity = read;
  ready = wait;
  guest = null;
}
export function aiOwner() {
  return identity()?.uid || "guest:" + guestScope();
}
function cancellable(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted)
    return Promise.reject(new DOMException("Cancelled", "AbortError"));
  return new Promise((resolve, reject) => {
    const cancel = () => {
      signal.removeEventListener("abort", cancel);
      reject(new DOMException("Cancelled", "AbortError"));
    };
    signal.addEventListener("abort", cancel, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", cancel);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", cancel);
        reject(error);
      },
    );
  });
}
export async function aiHeaders(signal) {
  await cancellable(Promise.resolve(ready()), signal);
  const user = identity();
  if (user?.getIdToken) {
    const token = await cancellable(user.getIdToken(), signal);
    if (identity()?.uid !== user.uid)
      throw new DOMException("Account changed", "AbortError");
    return { Authorization: "Bearer " + token };
  }
  if (!guest) {
    const controller = new AbortController(),
      timer = setTimeout(() => controller.abort(), 10000);
    guest = fetch("/api/session", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      signal: controller.signal,
    })
      .then(async (response) => {
        const result = await response.json().catch(() => ({}));
        if (!response.ok || typeof result.csrf !== "string") {
          const error = new Error(
            result.error ||
              "A secure AI session is unavailable. Your writing is kept.",
          );
          error.code = result.code;
          error.status = response.status;
          error.retryAfter =
            Number(response.headers?.get?.("retry-after")) || 0;
          throw error;
        }
        return result;
      })
      .catch((error) => {
        guest = null;
        throw error;
      })
      .finally(() => clearTimeout(timer));
  }
  const session = await guest;
  if (identity()?.uid) throw new DOMException("Account changed", "AbortError");
  if (session.expiresAt && session.expiresAt < Date.now()) {
    guest = null;
    return aiHeaders(signal);
  }
  return { "X-Classio-CSRF": session.csrf };
}
export async function authorizedFetch(url, options = {}) {
  if(options.signal?.aborted)throw new DOMException("Cancelled","AbortError");
  const headers = await cancellable(aiHeaders(options.signal), options.signal);
  const owner = aiOwner(),
    sessionUsed = guest;
  if (options.signal?.aborted)
    throw new DOMException("Cancelled", "AbortError");
  const send = (access) =>
    fetch(url, {
      ...options,
      credentials: "same-origin",
      headers: { ...options.headers, ...access },
    });
  const response = await send(headers);
  if (response.status !== 401) return response;
  const failure = await response
    .clone()
    .json()
    .catch(() => ({}));
  if (
    !["AUTH_REQUIRED", "AUTH_EXPIRED", "GUEST_SESSION_REQUIRED"].includes(
      failure.code,
    )
  )
    return response;
  if (options.signal?.aborted || aiOwner() !== owner)
    throw new DOMException(
      "Request cancelled or account changed",
      "AbortError",
    );
  let refreshed;
  if (headers.Authorization) {
    const user = identity();
    if (!user?.getIdToken || user.uid !== owner)
      throw new DOMException("Account changed", "AbortError");
    let pending = refreshing.get(owner);
    if (!pending) {
      pending = Promise.resolve()
        .then(() => user.getIdToken(true))
        .finally(() => {
          if (refreshing.get(owner) === pending) refreshing.delete(owner);
        });
      refreshing.set(owner, pending);
    }
    try {
      refreshed = {
        Authorization: "Bearer " + (await cancellable(pending, options.signal)),
      };
    } catch (error) {
      if (error.name === "AbortError") throw error;
      return response;
    }
  } else {
    if (guest === sessionUsed) guest = null;
    refreshed = await cancellable(aiHeaders(options.signal), options.signal);
  }
  if (options.signal?.aborted || aiOwner() !== owner)
    throw new DOMException(
      "Request cancelled or account changed",
      "AbortError",
    );
  // A rejected request has not reached inference. Replay JSON once with renewed
  // authorization; never retry an outage, revoked account or a second rejection.
  return send(refreshed);
}
