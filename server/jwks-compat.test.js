const { test, before, after } = require("node:test"),
  assert = require("node:assert/strict");
const http = require("node:http"),
  crypto = require("node:crypto");
const jwks = require("jwks-rsa");
const keys = crypto.generateKeyPairSync("rsa", { modulusLength: 2048 });
const publicJwk = {
  ...keys.publicKey.export({ format: "jwk" }),
  kid: "fixture-key",
  alg: "RS256",
  use: "sig",
};
let origin;
const server = http.createServer((req, res) => {
  res.setHeader("Content-Type", "application/json");
  res.setHeader("Connection", "close");
  const key =
    req.url === "/private"
      ? {
          ...keys.privateKey.export({ format: "jwk" }),
          kid: "fixture-key",
          alg: "RS256",
          use: "sig",
        }
      : publicJwk;
  res.end(JSON.stringify({ keys: [key] }));
});
before(async () => {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  origin = "http://127.0.0.1:" + server.address().port;
});
after(async () => {
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
});
test("the maintained jose implementation converts real public JWKS keys for signature verification", async () => {
  const client = jwks({ jwksUri: origin + "/public", timeout: 2000 });
  const key = await client.getSigningKey("fixture-key");
  const message = Buffer.from("synthetic signed study identity");
  const signature = crypto.sign("RSA-SHA256", message, keys.privateKey);
  assert(crypto.verify("RSA-SHA256", message, key.getPublicKey(), signature));
  assert(
    !crypto.verify(
      "RSA-SHA256",
      Buffer.from("forged identity"),
      key.getPublicKey(),
      signature,
    ),
  );
});
test("private JWKS material is rejected rather than used as a signing key", async () => {
  const client = jwks({ jwksUri: origin + "/private", timeout: 2000 });
  await assert.rejects(client.getSigningKey("fixture-key"));
});
test("the callback passport integration remains asynchronous and preserves invalid-token handling", async () => {
  const provider = jwks.passportJwtSecret({
    jwksUri: origin + "/public",
    timeout: 2000,
  });
  const body =
    Buffer.from(JSON.stringify({ alg: "RS256", kid: "fixture-key" })).toString(
      "base64url",
    ) +
    "." +
    Buffer.from(JSON.stringify({ sub: "fixture-student" })).toString(
      "base64url",
    );
  const token =
    body +
    "." +
    crypto
      .sign("RSA-SHA256", Buffer.from(body), keys.privateKey)
      .toString("base64url");
  const key = await new Promise((resolve, reject) => {
    assert.equal(
      provider({}, token, (error, value) =>
        error ? reject(error) : resolve(value),
      ),
      undefined,
    );
  });
  assert(
    crypto.verify(
      "RSA-SHA256",
      Buffer.from(body),
      key,
      Buffer.from(token.split(".")[2], "base64url"),
    ),
  );
  assert.equal(
    await new Promise((resolve, reject) =>
      provider({}, "invalid-token", (error, value) =>
        error ? reject(error) : resolve(value),
      ),
    ),
    null,
  );
});
