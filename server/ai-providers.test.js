const { test } = require("node:test");
const assert = require("node:assert/strict");
const { availableProviders, geminiPayload, generateStudy } = require("./ai-providers");

const env = { GROQ_API_KEY: "test-groq", GEMINI_API_KEY: "test-google", GEMINI_ENABLED: "true", GEMINI_MODEL: "test-model" };
const input = { messages: [{ role: "system", content: "Use the learner's language." }, { role: "user", content: "Explain gravity." }], maxTokens: 500 };

test("Gemini requires explicit enablement, a key and a valid configured model", () => {
  assert.deepEqual(availableProviders({ ...env, GEMINI_ENABLED: "false" }), ["groq"]);
  assert.deepEqual(availableProviders({ ...env, GEMINI_MODEL: "" }), ["groq"]);
  assert.deepEqual(availableProviders({ ...env, GEMINI_MODEL: "../other/key" }), ["groq"]);
  assert.deepEqual(availableProviders({ ...env, AI_PROVIDER_ORDER: "gemini,gemini,unknown,groq" }), ["gemini", "groq"]);
});

test("Gemini conversion preserves system instructions, conversation roles and image bytes", () => {
  const payload = geminiPayload({ ...input, messages: [...input.messages,
    { role: "assistant", content: "Gravity attracts masses." },
    { role: "user", content: [{ type: "image_url", image_url: { url: "data:image/png;base64,aGVsbG8=" } }, { type: "text", text: "Explain this diagram." }] },
  ] });
  assert.equal(payload.systemInstruction.parts[0].text, input.messages[0].content);
  assert.deepEqual(payload.contents.map(item => item.role), ["user", "model", "user"]);
  assert.deepEqual(payload.contents[2].parts[0], { inlineData: { mimeType: "image/png", data: "aGVsbG8=" } });
  assert.equal(payload.generationConfig.maxOutputTokens, 500);
});

test("Groq rate limiting falls back to Gemini without sending keys in URLs or returning thought text", async () => {
  const calls = [], reports = [];
  const result = await generateStudy(input, { env, report: item => reports.push(item), fetcher: async (url, options) => {
    calls.push({ url, options });
    if (calls.length === 1) return { ok: false, status: 429, json: async () => ({}) };
    return { ok: true, json: async () => ({ candidates: [{ finishReason: "STOP", content: { parts: [
      { thought: true, text: "Private reasoning" }, { text: "Gravity attracts masses." },
    ] } }] }) };
  } });
  assert.deepEqual(result, { provider: "gemini", text: "Gravity attracts masses." });
  assert.equal(calls.length, 2);
  assert.equal(calls[1].options.headers["x-goog-api-key"], env.GEMINI_API_KEY);
  assert(!calls[1].url.includes(env.GEMINI_API_KEY));
  assert.deepEqual(reports, [{ provider: "groq", code: "HTTP_429" }]);
});

test("safety-blocked responses do not bypass the block through another provider", async () => {
  let calls = 0;
  await assert.rejects(generateStudy(input, { env: { ...env, AI_PROVIDER_ORDER: "gemini,groq" }, fetcher: async () => {
    calls++;
    return { ok: true, json: async () => ({ promptFeedback: { blockReason: "SAFETY" } }) };
  } }), error => error.code === "CONTENT_BLOCKED" && error.status === 422);
  assert.equal(calls, 1);
});

test("network failures fail over, while invalid requests stop without repeating the request", async () => {
  let calls = 0;
  const result = await generateStudy(input, { env, fetcher: async () => {
    if (++calls === 1) throw new Error("network details that must stay private");
    return { ok: true, json: async () => ({ candidates: [{ content: { parts: [{ text: "A real provider response" }] } }] }) };
  } });
  assert.equal(result.provider, "gemini");
  calls = 0;
  await assert.rejects(generateStudy(input, { env, fetcher: async () => {
    calls++; return { ok: false, status: 400, json: async () => ({}) };
  } }), error => error.code === "HTTP_400");
  assert.equal(calls, 1);
});

test("Gemini accepts inline recordings and requests verbatim transcription", async () => {
  const audio = { audio: "dGVzdA==", filename: "lecture.webm" };
  const payload = geminiPayload(audio);
  assert.deepEqual(payload.contents[0].parts[1], { inlineData: { mimeType: "audio/webm", data: audio.audio } });
  assert.match(payload.contents[0].parts[0].text, /verbatim/);
});

test("unconfigured providers never make a network request or invent an answer", async () => {
  await assert.rejects(generateStudy(input, { env: {}, fetcher: () => { throw new Error("Unexpected request"); } }),
    error => error.code === "NOT_CONFIGURED" && error.status === 503);
});

test("truncated course/JSON output is reported instead of presented as a complete answer", async () => {
  await assert.rejects(generateStudy(input, { env: { GEMINI_ENABLED: "true", GEMINI_API_KEY: "test", GEMINI_MODEL: "test-model" },
    fetcher: async () => ({ ok: true, json: async () => ({ candidates: [{ finishReason: "MAX_TOKENS" }] }) }) }),
  error => error.code === "OUTPUT_LIMIT");
});
