const https = require("node:https"),
  http = require("node:http");
const { publicUrl, publicAddresses } = require("./public-source");
// Only server configuration selects an endpoint. Public addresses are DNS-pinned;
// redirects are rejected, so an authenticated endpoint cannot redirect a key.
async function ollamaFetch(path, options, env) {
  const base = new URL(env.OLLAMA_BASE_URL),
    url = new URL(base.href.replace(/\/$/, "") + path);
  const local =
    !env.VERCEL &&
    env.NODE_ENV !== "production" &&
    base.protocol === "http:" &&
    ["localhost", "127.0.0.1", "[::1]"].includes(base.hostname);
  let addresses;
  if (!local) {
    publicUrl(url.href);
    addresses = await publicAddresses(url);
  }
  if (url.origin !== base.origin || base.username || base.password)
    throw Error("ENDPOINT_REJECTED");
  return new Promise((resolve, reject) => {
    const body = options.body || "",
      transport = url.protocol === "https:" ? https : http;
    const request = transport.request(
      url,
      {
        method: options.method || "GET",
        headers: options.headers,
        signal: options.signal,
        ...(addresses
          ? {
              lookup: (_host, settings, callback) =>
                callback(
                  null,
                  settings?.all ? addresses : addresses[0].address,
                  settings?.all ? undefined : addresses[0].family,
                ),
            }
          : {}),
      },
      (response) => {
        let size = 0;
        const chunks = [];
        response.on("data", (chunk) => {
          size += chunk.length;
          if (size > 1500000) {
            response.destroy(Error("RESPONSE_TOO_LARGE"));
            return;
          }
          chunks.push(chunk);
        });
        response.on("error", reject);
        response.on("end", () =>
          resolve({
            ok: response.statusCode >= 200 && response.statusCode < 300,
            status: response.statusCode,
            headers: { get: (name) => response.headers[name.toLowerCase()] },
            json: async () =>
              JSON.parse(Buffer.concat(chunks).toString("utf8")),
          }),
        );
      },
    );
    request.on("error", reject);
    request.end(body);
  });
}
module.exports = { ollamaFetch };
