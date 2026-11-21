const { test } = require("node:test"),
  assert = require("node:assert/strict");
const { AccessError } = require("./ai-auth");
const { ImageServiceError } = require("./image-service");
let authFailure,
  operationFailure,
  quotaReady = false,
  quota = {};
require.cache[require.resolve("./ai-auth")].exports = {
  AccessError,
  authorize: async () => {
    if (authFailure) throw authFailure;
    return { sub: "user:fixture", kind: "account" };
  },
};
require.cache[require.resolve("./ai-quota")] = {
  exports: { configured: () => quotaReady, reserveRequest: async () => quota },
};
require.cache[require.resolve("./image-service")].exports = {
  ImageServiceError,
  imageCapabilities: () => ({ imageGeneration: true, backgroundRemoval: true }),
  imageOperation: async () => {
    if (operationFailure) throw operationFailure;
    return { image: "data:image/png;base64,fixture" };
  },
};
const images = require("../api/images");
async function invoke(method = "POST") {
  const result = { headers: {} };
  await images(
    {
      method,
      headers: {
        "content-type": "application/json",
        "sec-fetch-site": "same-origin",
      },
      body: { action: "generate", prompt: "A classroom leaf" },
    },
    {
      setHeader: (name, value) => (result.headers[name] = value),
      status(code) {
        result.status = code;
        return this;
      },
      json(body) {
        result.body = body;
      },
    },
  );
  return result;
}
test("image-service errors keep their operation code instead of being mislabeled as sign-in outages", async () => {
  operationFailure = new ImageServiceError(
    503,
    "Image generation is unavailable.",
    "IMAGE_NOT_CONFIGURED",
  );
  let result = await invoke();
  assert.equal(result.status, 503);
  assert.equal(result.body.code, "IMAGE_NOT_CONFIGURED");
  operationFailure = new ImageServiceError(
    429,
    "The free image service is busy.",
    "IMAGE_RATE_LIMIT",
  );
  operationFailure.retryAfter = 90;
  result = await invoke();
  assert.equal(result.status, 429);
  assert.equal(result.headers["Retry-After"], "90");
  assert.equal(result.body.code, "IMAGE_RATE_LIMIT");
  operationFailure = undefined;
});
test("authentication errors remain protected and unknown SDK errors never expose private details", async () => {
  authFailure = new AccessError(401, "AUTH_REVOKED");
  let result = await invoke();
  assert.equal(result.body.code, "AUTH_REVOKED");
  authFailure = undefined;
  operationFailure = Object.assign(
    Error("private provider token and response"),
    { code: "untrusted-code", status: 400 },
  );
  result = await invoke();
  assert.equal(result.status, 503);
  assert(!JSON.stringify(result).includes("private"));
  assert(!JSON.stringify(result).includes("untrusted-code"));
  operationFailure = undefined;
});

test("public image capabilities require the shared quota and secure session configuration", async () => {
  const before = process.env.CLASSIO_SESSION_SECRET;
  try {
    process.env.CLASSIO_SESSION_SECRET =
      "fixture-image-session-secret-at-least-32-characters";
    quotaReady = false;
    let result = await invoke("GET");
    assert.equal(result.body.imageGeneration, false);
    assert.equal(result.body.backgroundRemoval, false);
    quotaReady = true;
    result = await invoke("GET");
    assert.equal(result.body.imageGeneration, true);
    delete process.env.CLASSIO_SESSION_SECRET;
    result = await invoke("GET");
    assert.equal(result.body.imageGeneration, false);
  } finally {
    quotaReady = false;
    if (before === undefined) delete process.env.CLASSIO_SESSION_SECRET;
    else process.env.CLASSIO_SESSION_SECRET = before;
  }
});
