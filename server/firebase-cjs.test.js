const { test } = require("node:test"),
  assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process"),
  path = require("node:path");
test("Firebase JWT verification and JWKS integrations work with synchronous require(ESM) disabled", () => {
  const env = { ...process.env };
  // Start an independent runner, rather than inheriting the parent's worker mode.
  delete env.NODE_TEST_CONTEXT;
  const result = spawnSync(
    process.execPath,
    [
      "--no-experimental-require-module",
      "--test",
      "--test-reporter=tap",
      "server/firebase-auth-crypto.test.js",
      "server/jwks-compat.test.js",
    ],
    {
      cwd: path.resolve(__dirname, ".."),
      encoding: "utf8",
      timeout: 20000,
      env,
    },
  );
  assert.equal(
    result.status,
    0,
    result.error?.message || result.stdout + result.stderr,
  );
  assert.match(result.stdout, /# tests [1-9]/);
});
