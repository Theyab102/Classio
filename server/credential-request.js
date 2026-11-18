const { providerCredentials } = require("./ai-settings");
// Rotate only a rejected credential; quota, billing and outage errors do not
// become a second request against the same account's allowance.
async function credentialRequest(
  provider,
  build,
  {
    env = process.env,
    fetcher = global.fetch,
    signal = AbortSignal.timeout(12000),
  } = {},
) {
  let response;
  for (const credential of providerCredentials(provider, env)) {
    const { url, ...options } = build(credential);
    response = await fetcher(url, { ...options, signal, redirect: "error" });
    if (![401, 403].includes(response.status)) return response;
    await response.body?.cancel?.().catch(() => {});
  }
  return response;
}
module.exports = { credentialRequest };
