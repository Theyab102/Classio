const {
  availableProviders,
  providerHealth,
  credentialHealth,
} = require("./ai-providers");
const { configured: quotaConfigured, quotaMode } = require("./ai-quota");
const { providers, providerPolicy, AUDIENCE } = require("./ai-policy");
const { configuredNames, credentialStatus } = require("./ai-settings");
function configurationStatus(env = process.env) {
  const enabled = availableProviders(env),
    missing = [];
  if ((env.CLASSIO_SESSION_SECRET || "").length < 32)
    missing.push("CLASSIO_SESSION_SECRET (at least 32 characters)");
  if (!enabled.length)
    missing.push(
      "An eligible free provider, its primary credential, and server-side eligibility confirmations",
    );
  if (!quotaConfigured(env)) {
    if (quotaMode(env) === "firestore")
      missing.push(
        "FIREBASE_SERVICE_ACCOUNT_JSON and CLASSIO_SESSION_SECRET (or AI_RATE_LIMIT_SECRET)",
      );
    else missing.push("AI_QUOTA_MODE must be instance or firestore");
  }
  return {
    configured: !missing.length,
    providers: enabled,
    missing,
    settingNames: configuredNames(env),
    quotaMode: quotaMode(env),
    audience: AUDIENCE,
    freeOnly: env.AI_ALLOW_PAID !== "true",
    providerConfiguration: Object.fromEntries(
      providers.map((provider) => [
        provider,
        {
          ...credentialStatus(provider, env),
          credentials: credentialHealth(env)[provider],
          ...providerPolicy(provider, env),
          health: providerHealth()[provider]?.status || "available",
        },
      ]),
    ),
    browserLocalEnabled: env.AI_ENABLE_BROWSER_LOCAL === "true",
  };
}
function publicStatus(env = process.env) {
  const report = configurationStatus(env);
  return {
    aiConfigured: report.configured,
    providers: report.providers.map((id) => ({
      id,
      status: providerHealth()[id]?.status || "available",
    })),
    freeOnly: report.freeOnly,
    browserLocalEnabled: report.browserLocalEnabled,
  };
}
module.exports = { configurationStatus, publicStatus };
