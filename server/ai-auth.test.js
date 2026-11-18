const { test } = require("node:test"),
  assert = require("node:assert/strict");
const { authorize, createGuestSession, readSession } = require("./ai-auth");
const { parseStudyRequest } = require("./study-request");
const env = {
  CLASSIO_SESSION_SECRET: "fixture-session-secret-at-least-32-characters",
};
function guest() {
  let cookie;
  const body = createGuestSession(
    {
      headers: {
        host: "classio.example",
        origin: "https://classio.example",
        "sec-fetch-site": "same-origin",
      },
    },
    {
      setHeader(_name, value) {
        cookie = value;
      },
    },
    env,
  );
  return {
    body,
    cookie,
    req: {
      headers: {
        host: "classio.example",
        cookie: cookie.split(";")[0],
        "x-classio-csrf": body.csrf,
      },
    },
  };
}
test("signed guest session uses secure HttpOnly cookie, CSRF, bounded lifetime and unforgeable identity", async () => {
  const session = guest();
  assert.match(session.cookie, /HttpOnly; Secure; SameSite=Strict/);
  assert(!session.body.sub);
  assert(!JSON.stringify(session.body).includes(env.CLASSIO_SESSION_SECRET));
  assert.equal((await authorize(session.req, { env })).kind, "guest");
  for (const headers of [
    {},
    { cookie: session.req.headers.cookie },
    { ...session.req.headers, "x-classio-csrf": "wrong" },
    { ...session.req.headers, cookie: session.req.headers.cookie + "extra" },
    { ...session.req.headers, "x-classio-csrf": "é".repeat(43) },
  ])
    await assert.rejects(
      authorize({ headers }, { env }),
      (error) => error.status === 401,
    );
  assert.equal(readSession(session.req, env, Date.now() + 13 * 3600000), null);
});
test("cross-origin session creation/use is rejected independently of credentials", async () => {
  const session = guest();
  for (const headers of [
    { ...session.req.headers, "sec-fetch-site": "cross-site" },
    { ...session.req.headers, origin: "https://evil.example" },
  ])
    await assert.rejects(
      authorize({ headers }, { env }),
      (error) => error.status === 403,
    );
});
test("account requests require verified Google Firebase identity; forged bearer tokens never become guests", async () => {
  const req = {
    headers: { authorization: "Bearer fixture-token-at-least-20-chars" },
  };
  assert.deepEqual(
    await authorize(req, {
      env,
      verify: async () => ({
        uid: "alice",
        firebase: { sign_in_provider: "google.com" },
      }),
    }),
    { sub: "user:alice", kind: "account" },
  );
  await assert.rejects(
    authorize(req, {
      env,
      verify: async () => {
        throw Object.assign(Error("private token details"), {
          code: "auth/invalid-id-token",
        });
      },
    }),
    (error) => error.status === 401 && !error.message.includes("private"),
  );
  await assert.rejects(
    authorize(req, {
      env,
      verify: async () => ({
        uid: "alice",
        firebase: { sign_in_provider: "anonymous" },
      }),
    }),
    (error) => error.status === 403,
  );
});
test("guest signing fails closed without a server secret and student payload cannot select endpoints or secrets", () => {
  assert.throws(
    () => createGuestSession({ headers: {} }, { setHeader() {} }, {}),
    (error) => error.status === 503,
  );
  const parsed = parseStudyRequest({
    messages: [{ role: "user", content: "Explain gravity" }],
    maxTokens: 500,
    provider: "groq",
    baseUrl: "https://evil.example",
    apiKey: "fixture",
    allowPaid: true,
  });
  assert.equal(parsed.provider, "groq");
  assert(!parsed.baseUrl);
  assert(!parsed.apiKey);
  assert(!parsed.allowPaid);
  assert.equal(
    parseStudyRequest({
      messages: [{ role: "user", content: "Hello" }],
      maxTokens: 500,
      provider: "https://evil.example",
    }),
    null,
  );
});

test("server configuration and certificate outages are not reported as a student sign-in rejection", async () => {
  const req = {
    headers: { authorization: "Bearer fixture-token-at-least-20-chars" },
  };
  for (const error of [
    Object.assign(Error("credential failure"), {
      code: "auth/insufficient-permission",
    }),
    Object.assign(
      Error("Error fetching public keys for Google certs: private response"),
      { code: "auth/argument-error" },
    ),
    Object.assign(Error("network failure"), { code: "app/network-error" }),
  ])
    await assert.rejects(
      authorize(req, {
        env,
        verify: async () => {
          throw error;
        },
      }),
      (result) => result.status === 503 && !result.message.includes("private"),
    );
  await assert.rejects(
    authorize(req, {
      env,
      verify: async () => {
        throw Object.assign(Error("Expired"), {
          code: "auth/id-token-expired",
        });
      },
    }),
    (error) => error.status === 401 && error.code === "AUTH_EXPIRED",
  );
});
