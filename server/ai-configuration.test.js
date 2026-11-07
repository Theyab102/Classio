const { test } = require("node:test");
const assert = require("node:assert/strict");
const { configurationStatus } = require("./ai-configuration");
const { reserveInstance, configured } = require("./ai-quota");

test("accepts the confirmed server aliases without requiring a Firebase administrator", () => {
  const report = configurationStatus({ Groq_Key: "test-provider-secret", Gemini_Key: "test-google-secret" });
  assert.equal(report.configured, true); assert.deepEqual(report.providers, ["groq"]);
  assert.deepEqual(report.settingNames, ["Groq_Key", "Gemini_Key"]); assert.equal(report.quotaMode, "instance");
  assert.equal(report.geminiOptInRequired, true);
  assert(!JSON.stringify(report).includes("test-provider-secret")); assert(!JSON.stringify(report).includes("test-google-secret"));
});
test("never accepts public build variables, placeholder values, or a disabled Gemini key alone", () => {
  assert(!configurationStatus({ REACT_APP_GROQ_KEY: "test-key" }).configured);
  assert(!configurationStatus({ Groq_Key: "Your_new_Groq_API_key" }).configured);
  assert(!configurationStatus({ Groq_Key: "gsk_example..." }).configured);
  assert(!configurationStatus({ Gemini_Key: "test-google-secret" }).configured);
  assert(configurationStatus({ Gemini_Key: "test-google-secret", GEMINI_ENABLED: "true" }).configured);
});
test("shared quota storage fails closed when explicitly selected but misconfigured", () => {
  assert(!configured({ AI_QUOTA_MODE: "firestore" }));
  assert(!configured({ AI_QUOTA_MODE: "invalid" }));
  assert(configured({ FIREBASE_SERVICE_ACCOUNT_JSON: JSON.stringify({ project_id: "test", client_email: "test@example.com", private_key: "fixture" }), AI_RATE_LIMIT_SECRET: "test-only-secret-at-least-32-characters" }));
});
test("instance limiter enforces client and instance limits and expires counters", () => {
  const store = new Map(), now = 1;
  for (let i = 0; i < 18; i++) assert.deepEqual(reserveInstance("study", "client-a", now, store), {});
  assert(reserveInstance("study", "client-a", now, store).retryAfter > 0);
  assert.deepEqual(reserveInstance("study", "client-a", 600001, store), {});
  for (let i = 0; i < 231; i++) assert.deepEqual(reserveInstance("study", `other-${i}`, 600001, store), {});
  assert(reserveInstance("study", "final", 600001, store).retryAfter > 0);
  assert.deepEqual(reserveInstance("study", "final", 86400001, store), {});
});
test("audio limits are separate and malformed quota mode is diagnosed by name", () => {
  const store = new Map(); for (let i = 0; i < 3; i++) assert.deepEqual(reserveInstance("audio", "client", 1, store), {});
  assert(reserveInstance("audio", "client", 1, store).retryAfter > 0); assert.deepEqual(reserveInstance("study", "client", 1, store), {});
  assert.deepEqual(configurationStatus({ Groq_Key: "fixture", AI_QUOTA_MODE: "bad" }).missing, ["AI_QUOTA_MODE must be instance or firestore"]);
});
