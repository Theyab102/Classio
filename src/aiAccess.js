import { guestScope } from "./guestWorkspace";
let identity = () => null,
  guest = null;
export function setAiIdentityProvider(read) {
  identity = read;
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
  const user = identity();
  if (user?.getIdToken)
    return { Authorization: "Bearer " + (await user.getIdToken()) };
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
  if (session.expiresAt && session.expiresAt < Date.now()) {
    guest = null;
    return aiHeaders(signal);
  }
  return { "X-Classio-CSRF": session.csrf };
}
export async function authorizedFetch(url, options = {}) {
  const headers = await cancellable(aiHeaders(options.signal), options.signal);
  if (options.signal?.aborted)
    throw new DOMException("Cancelled", "AbortError");
  const response = await fetch(url, {
    ...options,
    credentials: "same-origin",
    headers: { ...options.headers, ...headers },
  });
  if (response.status === 401) guest = null;
  return response;
}
