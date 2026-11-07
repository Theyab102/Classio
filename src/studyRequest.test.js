import { sendStudyRequest } from "./studyTools";

test("a stalled study request times out and clears its timer", async () => {
  jest.useFakeTimers();
  const previousFetch = global.fetch;
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
  }));
  try {
    const pending = sendStudyRequest({ messages: [{ role: "user", content: "Explain gravity" }], maxTokens: 100 });
    const result = expect(pending).rejects.toThrow("took too long");
    jest.advanceTimersByTime(55000);
    await result;
    expect(jest.getTimerCount()).toBe(0);
  } finally { global.fetch = previousFetch; jest.useRealTimers(); }
});

test("an interrupted tool request preserves cancellation instead of displaying a provider error", async () => {
  const previousFetch = global.fetch;
  const controller = new AbortController();
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => {
    options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
  }));
  try {
    const pending = sendStudyRequest({}, controller.signal);
    controller.abort();
    await expect(pending).rejects.toHaveProperty("name", "AbortError");
  } finally { global.fetch = previousFetch; }
});

test("provider rate-limit responses retain their code and retry interval", async () => {
  const previousFetch = global.fetch;
  global.fetch = jest.fn(async () => ({ ok: false, status: 429, headers: { get: () => "12" }, json: async () => ({ error: "Please wait before retrying.", code: "PROVIDER_RATE_LIMIT" }) }));
  try { await expect(sendStudyRequest({})).rejects.toMatchObject({ status: 429, code: "PROVIDER_RATE_LIMIT", retryAfter: 12, message: "Please wait before retrying." }); }
  finally { global.fetch = previousFetch; }
});
