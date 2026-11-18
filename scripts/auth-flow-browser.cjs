// Real HTTP routes, secure guest cookies, CSRF and Firebase JWT signatures.
// The remote certificate transport and AI response are explicit fixtures.
const fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  http = require("node:http"),
  crypto = require("node:crypto"),
  assert = require("node:assert/strict"),
  webpack = require("webpack");
const { chromium } = require("playwright");
const root = path.resolve(__dirname, ".."),
  temp = fs.mkdtempSync(path.join(os.tmpdir(), "classio-auth-flow-"));
const fixture = {
  CLASSIO_SESSION_SECRET: "fixture-session-secret-at-least-32-characters",
  AI_QUOTA_MODE: "instance",
  Groq_Key: "fixture-provider-key",
  AI_GROQ_STUDENT_USE_APPROVED: "true",
  AI_GROQ_FREE_TIER_CONFIRMED: "true",
};
for (const name of [
  "VERCEL",
  "FIREBASE_SERVICE_ACCOUNT_JSON",
  "FIREBASE_AUTH_EMULATOR_HOST",
])
  delete process.env[name];
Object.assign(process.env, fixture);
const { initializeApp, deleteApp } = require("firebase-admin/app"),
  { getAuth } = require("firebase-admin/auth"),
  app = initializeApp(
    { projectId: "classio-4378f" },
    "classio-auth:classio-4378f",
  ),
  auth = getAuth(app),
  keys = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
auth.idTokenVerifier.signatureVerifier.keyFetcher.fetchPublicKeys =
  async () => ({
    "fixture-key": keys.publicKey.export({ type: "spki", format: "pem" }),
  });
function jwt(expired = false) {
  const now = Math.floor(Date.now() / 1000),
    payload = {
      aud: "classio-4378f",
      iss: "https://securetoken.google.com/classio-4378f",
      sub: "fixture-account",
      iat: now - 60,
      exp: expired ? now - 10 : now + 3600,
      auth_time: now - 60,
      firebase: { sign_in_provider: "google.com" },
    },
    body =
      Buffer.from(
        JSON.stringify({ alg: "RS256", kid: "fixture-key" }),
      ).toString("base64url") +
      "." +
      Buffer.from(JSON.stringify(payload)).toString("base64url");
  return (
    body +
    "." +
    crypto
      .sign("RSA-SHA256", Buffer.from(body), keys.privateKey)
      .toString("base64url")
  );
}
let generated = 0;
const transport = global.fetch;
global.fetch = async (url, options) => {
  if (String(url).startsWith("https://api.groq.com/")) {
    if (String(url).endsWith("/models"))
      return new Response(
        JSON.stringify({
          data: [{ id: "llama-3.1-8b-instant", context_window: 131072 }],
        }),
        { status: 200 },
      );
    generated++;
    return new Response(
      JSON.stringify({
        choices: [
          {
            message: { content: "Fixture completed content" },
            finish_reason: "stop",
          },
        ],
      }),
      { status: 200 },
    );
  }
  return transport(url, options);
};
const study = require("../api/study"),
  session = require("../api/session"),
  requests = [];
const server = http.createServer(async (req, res) => {
  res.status = (code) => {
    res.statusCode = code;
    return res;
  };
  res.json = (body) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(body));
  };
  try {
    if (req.url === "/harness.js") {
      res.setHeader("Content-Type", "application/javascript");
      return res.end(fs.readFileSync(path.join(temp, "harness.js")));
    }
    if (req.method === "GET") {
      res.setHeader("Content-Type", "text/html");
      return res.end(
        '<!doctype html><html><body><script src="/harness.js"></script></body></html>',
      );
    }
    let data = "";
    for await (const chunk of req) {
      data += chunk;
      if (data.length > 4200000) throw Error("Oversized fixture");
    }
    req.body = JSON.parse(data || "{}");
    requests.push({
      url: req.url,
      body: data,
      authorization: !!req.headers.authorization,
    });
    if (req.url === "/api/session") return await session(req, res);
    if (req.url === "/api/study") return await study(req, res);
    res.status(404).json({ error: "Unknown route" });
  } catch {
    res.status(500).json({ error: "Local verification server failed" });
  }
});
(async () => {
  let browser;
  try {
    const entry = path.join(temp, "entry.js");
    fs.writeFileSync(
      entry,
      `import{authorizedFetch,setAiIdentityProvider}from ${JSON.stringify(path.join(root, "src/aiAccess.js"))};window.classioAuthCheck={authorizedFetch,setAiIdentityProvider};`,
    );
    await new Promise((resolve, reject) =>
      webpack(
        {
          mode: "development",
          devtool: false,
          entry,
          output: { path: temp, filename: "harness.js" },
          resolve: { extensions: [".js"] },
          optimization: { minimize: false },
        },
        (error, stats) =>
          error
            ? reject(error)
            : stats.hasErrors()
              ? reject(Error(stats.toString({ all: false, errors: true })))
              : resolve(),
      ),
    );
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const origin = "http://127.0.0.1:" + server.address().port;
    const executable =
      process.env.CLASSIO_CHROMIUM_PATH ||
      (fs.existsSync("/usr/bin/chromium") ? "/usr/bin/chromium" : undefined);
    browser = await chromium.launch({
      executablePath: executable,
      headless: true,
      args: ["--no-sandbox"],
    });
    const context = await browser.newContext(),
      page = await context.newPage();
    await page.goto(origin);
    await page.waitForFunction(() => !!window.classioAuthCheck);
    const payload = {
      messages: [{ role: "user", content: "Explain projectile motion." }],
      maxTokens: 500,
      task: "notes",
    };
    const run = (body) =>
      page.evaluate(async (body) => {
        try {
          const reply = await window.classioAuthCheck.authorizedFetch(
            "/api/study",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body),
            },
          );
          return { status: reply.status, data: await reply.json() };
        } catch (error) {
          return {
            status: error.status || 0,
            data: { code: error.code, error: error.message },
          };
        }
      }, body);
    await page.evaluate(
      ({ oldToken, freshToken }) => {
        window.__refreshes = 0;
        let current = oldToken;
        window.classioAuthCheck.setAiIdentityProvider(() => ({
          uid: "fixture-account",
          getIdToken: async (force) => {
            if (force) {
              window.__refreshes++;
              current = freshToken;
            }
            return current;
          },
        }));
      },
      { oldToken: jwt(true), freshToken: jwt() },
    );
    const first = await run(payload);
    assert.equal(first.status, 200);
    assert.equal(await page.evaluate(() => window.__refreshes), 1);
    assert.equal(generated, 1);
    assert.deepEqual(
      requests.filter((r) => r.url === "/api/study").map((r) => r.body),
      [JSON.stringify(payload), JSON.stringify(payload)],
    );
    for (const task of ["presentation", "flashcards", "conversation"])
      assert.equal((await run({ ...payload, task })).status, 200);
    assert.equal(await page.evaluate(() => window.__refreshes), 1);
    console.log(
      "PASS account: real expired Firebase token refreshed once; notes/slides/cards/podcast questions reach their actual route with no duplicate generation.",
    );
    await page.evaluate(() =>
      window.classioAuthCheck.setAiIdentityProvider(() => null),
    );
    assert.equal((await run({ ...payload, task: "presentation" })).status, 200);
    const cookies = await context.cookies();
    assert(
      cookies.some(
        (cookie) =>
          cookie.name === "__Host-classio_ai" &&
          cookie.httpOnly &&
          cookie.secure,
      ),
      "Signed secure guest cookie was not stored by the browser",
    );
    await context.clearCookies();
    const before = generated;
    assert.equal((await run({ ...payload, task: "flashcards" })).status, 200);
    assert.equal(generated, before + 1);
    console.log(
      "PASS guest: real cookie/CSRF session renewed after cookie loss; request generated once.",
    );
    const missing = await transport(origin + "/api/study", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    assert.equal(missing.status, 401);
    delete process.env.CLASSIO_SESSION_SECRET;
    await context.clearCookies();
    await page.evaluate(() =>
      window.classioAuthCheck.setAiIdentityProvider(() => null),
    );
    const broken = await run(payload);
    assert.equal(broken.status, 503);
    assert.equal(broken.data.code, "AI_NOT_CONFIGURED");
    assert(!/sign in again/.test(broken.data.error));
    console.log(
      "PASS configuration: missing server setup fails closed with a service message, without blaming the student login.",
    );
    await context.close();
  } finally {
    await browser?.close();
    await new Promise((resolve) =>
      server.listening ? server.close(resolve) : resolve(),
    );
    global.fetch = transport;
    await deleteApp(app);
    fs.rmSync(temp, { recursive: true, force: true });
  }
})().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
