import { authorizedFetch, setAiIdentityProvider } from "./aiAccess";
const response = (status, body = {}) => ({
  status,
  ok: status < 400,
  headers: { get: () => null },
  json: async () => body,
  clone: () => response(status, body),
});
const post = {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: '{"task":"notes","source":"Keep source unchanged"}',
};
afterEach(() => {
  setAiIdentityProvider(() => null);
  jest.restoreAllMocks();
});
test("an expired account token refreshes once and resends the exact study payload", async () => {
  const user = {
    uid: "alice",
    getIdToken: jest.fn(async (force) =>
      force ? "fixture-fresh-token" : "fixture-old-token",
    ),
  };
  setAiIdentityProvider(() => user);
  global.fetch = jest
    .fn()
    .mockResolvedValueOnce(response(401, { code: "AUTH_EXPIRED" }))
    .mockResolvedValueOnce(response(200, { text: "Study notes" }));
  expect((await authorizedFetch("/api/study", post)).status).toBe(200);
  expect(user.getIdToken.mock.calls).toEqual([[], [true]]);
  expect(fetch.mock.calls.map(([, options]) => options.body)).toEqual([
    post.body,
    post.body,
  ]);
  expect(fetch.mock.calls[1][1].headers.Authorization).toBe(
    "Bearer fixture-fresh-token",
  );
});
test("a second rejection stops; configuration outages and revoked accounts never become guest requests", async () => {
  const user = {
    uid: "alice",
    getIdToken: jest.fn(async () => "fixture-token"),
  };
  setAiIdentityProvider(() => user);
  for (const [status, code, count] of [
    [401, "AUTH_EXPIRED", 2],
    [503, "AUTH_UNAVAILABLE", 1],
    [503, "AI_NOT_CONFIGURED", 1],
    [401, "AUTH_REVOKED", 1],
  ]) {
    global.fetch = jest.fn(async () => response(status, { code }));
    expect((await authorizedFetch("/api/study", post)).status).toBe(status);
    expect(fetch).toHaveBeenCalledTimes(count);
    expect(fetch.mock.calls.every(([url]) => url === "/api/study")).toBe(true);
  }
});
test("an expired guest cookie renews its CSRF session once and preserves the generation request", async () => {
  setAiIdentityProvider(() => null);
  let sessions = 0,
    attempts = 0;
  global.fetch = jest.fn(async (url) =>
    url === "/api/session"
      ? response(200, {
          csrf: "fixture-csrf-" + ++sessions,
          expiresAt: Date.now() + 3600000,
        })
      : ++attempts === 1
        ? response(401, { code: "GUEST_SESSION_REQUIRED" })
        : response(200, { text: "Cards" }),
  );
  expect((await authorizedFetch("/api/study", post)).status).toBe(200);
  expect(sessions).toBe(2);
  const calls = fetch.mock.calls.filter(([url]) => url === "/api/study");
  expect(calls).toHaveLength(2);
  expect(calls[1][1].headers["X-Classio-CSRF"]).toBe("fixture-csrf-2");
  expect(calls[1][1].body).toBe(post.body);
});
test("a rejected request cannot replay under a different signed-in identity", async () => {
  let owner = {
    uid: "alice",
    getIdToken: jest.fn(async () => "fixture-token"),
  };
  setAiIdentityProvider(() => owner);
  global.fetch = jest.fn(async () => {
    owner = { uid: "bob", getIdToken: jest.fn() };
    return response(401, { code: "AUTH_EXPIRED" });
  });
  await expect(authorizedFetch("/api/study", post)).rejects.toMatchObject({
    name: "AbortError",
  });
  expect(fetch).toHaveBeenCalledTimes(1);
});
test("requests wait for Firebase initialization instead of issuing an accidental guest session", async () => {
  let owner = null,
    finish;
  const ready = new Promise(
    (resolve) =>
      (finish = () => {
        owner = {
          uid: "alice",
          getIdToken: async () => "fixture-initialized-token",
        };
        resolve();
      }),
  );
  setAiIdentityProvider(
    () => owner,
    () => ready,
  );
  global.fetch = jest.fn(async () => response(200, { text: "Ready" }));
  const pending = authorizedFetch("/api/study", post);
  await Promise.resolve();
  expect(fetch).not.toHaveBeenCalled();
  finish();
  await pending;
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(fetch.mock.calls[0][1].headers.Authorization).toBe(
    "Bearer fixture-initialized-token",
  );
});
