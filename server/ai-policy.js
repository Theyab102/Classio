const { providerCredentials } = require("./ai-settings");
const AUDIENCE = "school_students_13_plus";
const providers = ["groq", "gemini", "cloudflare", "openrouter", "ollama"];
function providerPolicy(provider, env = process.env) {
  const credentials = providerCredentials(provider, env),
    paid = env.AI_ALLOW_PAID === "true";
  if (provider === "gemini")
    return { enabled: false, reason: "AUDIENCE_RESTRICTED" }; // Developer API must not serve this minor audience.
  if (provider === "ollama" && env.AI_ENABLE_OLLAMA !== "true")
    return { enabled: false, reason: "NOT_ENABLED" };
  if (provider !== "ollama" && !credentials.length)
    return { enabled: false, reason: "NOT_CONFIGURED" };
  if (env["AI_" + provider.toUpperCase() + "_STUDENT_USE_APPROVED"] !== "true")
    return { enabled: false, reason: "ELIGIBILITY_REVIEW_REQUIRED" };
  if (
    provider === "cloudflare" &&
    !credentials.some((key) => /^[a-f\d]{32}$/i.test(key.accountId || ""))
  )
    return { enabled: false, reason: "ACCOUNT_CONFIGURATION_REQUIRED" };
  if (provider === "ollama") {
    let url;
    try {
      url = new URL(env.OLLAMA_BASE_URL);
    } catch {
      return { enabled: false, reason: "ENDPOINT_REQUIRED" };
    }
    const local =
      !env.VERCEL &&
      env.NODE_ENV !== "production" &&
      url.protocol === "http:" &&
      ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname);
    if (
      url.username ||
      url.password ||
      url.search ||
      url.hash ||
      (!local && (url.protocol !== "https:" || !credentials.length))
    )
      return { enabled: false, reason: "SECURE_ENDPOINT_REQUIRED" };
    if (!env.OLLAMA_MODEL || env.AI_OLLAMA_LICENSE_APPROVED !== "true")
      return { enabled: false, reason: "MODEL_REVIEW_REQUIRED" };
    if (!paid && env.AI_OLLAMA_FREE_TIER_CONFIRMED !== "true")
      return { enabled: false, reason: "FREE_OPERATION_UNVERIFIED" };
  }
  if (
    !paid &&
    ["groq", "cloudflare"].includes(provider) &&
    env["AI_" + provider.toUpperCase() + "_FREE_TIER_CONFIRMED"] !== "true"
  )
    return { enabled: false, reason: "FREE_OPERATION_UNVERIFIED" };
  return { enabled: true, reason: "AVAILABLE" };
}
function eligibleProviders(env = process.env) {
  return providers.filter((provider) => providerPolicy(provider, env).enabled);
}
module.exports = { AUDIENCE, providers, providerPolicy, eligibleProviders };
