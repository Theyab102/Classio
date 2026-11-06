const { test, after } = require("node:test");
const assert = require("node:assert/strict");
const { configurationStatus } = require("./ai-configuration");
const status = require("../api/status");

const fixture = {
  GROQ_API_KEY: "test-provider-secret",
  FIREBASE_SERVICE_ACCOUNT_JSON: JSON.stringify({ project_id: "test", client_email: "test@example.com", private_key: "test-private-key" }),
  AI_RATE_LIMIT_SECRET: "test-only-rate-secret-over-32-characters",
};

test("reports missing or invalid settings by name, without leaking credentials", () => {
  assert.deepEqual(configurationStatus({}), { configured: false, providers: [],
    missing: ["GROQ_API_KEY", "FIREBASE_SERVICE_ACCOUNT_JSON", "AI_RATE_LIMIT_SECRET"] });
  const invalid = configurationStatus({ ...fixture, FIREBASE_SERVICE_ACCOUNT_JSON: "{}", AI_RATE_LIMIT_SECRET: "short" });
  assert.deepEqual(invalid.missing, ["FIREBASE_SERVICE_ACCOUNT_JSON", "AI_RATE_LIMIT_SECRET"]);
  const report = configurationStatus(fixture);
  assert.deepEqual(report, { configured: true, providers: ["groq"], missing: [] });
  for (const secret of Object.values(fixture)) assert(!JSON.stringify(report).includes(secret));
});

test("Gemini remains opt-in and requires a model as well as a key", () => {
  const env = { ...fixture, GROQ_API_KEY: "", GEMINI_API_KEY: "test-google-secret", GEMINI_MODEL: "example-model" };
  assert(!configurationStatus(env).configured);
  assert(configurationStatus({ ...env, GEMINI_ENABLED: "true" }).configured);
  assert.deepEqual(configurationStatus({ ...env, GEMINI_ENABLED: "true", GEMINI_MODEL: "" }).missing, ["GEMINI_MODEL"]);
});

const names = [...Object.keys(fixture), "GEMINI_ENABLED", "GEMINI_API_KEY", "GEMINI_MODEL"];
const previous = Object.fromEntries(names.map(key => [key, process.env[key]]));
after(() => { for (const [key, value] of Object.entries(previous)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } });
function invoke(method) {
  const result = { headers: {} };
  status({ method }, {
    setHeader(key, value) { result.headers[key] = value; },
    status(code) { result.status = code; return this; },
    json(body) { result.body = body; }, end() { result.ended = true; },
  });
  return result;
}

test("deployment diagnostics distinguish configuration from live access and support HEAD", () => {
  names.forEach(key => delete process.env[key]);
  assert.equal(invoke("GET").status, 503);
  Object.assign(process.env, fixture);
  const report = invoke("GET");
  assert.equal(report.status, 200);
  assert.equal(report.headers["Cache-Control"], "no-store");
  assert.equal(report.body.aiConfigured, true);
  assert.match(report.body.checked, /live provider access is not tested/);
  for (const secret of Object.values(fixture)) assert(!JSON.stringify(report.body).includes(secret));
  const head = invoke("HEAD");
  assert.equal(head.status, 200);
  assert.equal(head.ended, true);
  assert.equal(head.body, undefined);
  assert.equal(invoke("POST").status, 405);
});
