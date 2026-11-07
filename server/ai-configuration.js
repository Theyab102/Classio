const { availableProviders } = require("./ai-providers");
const { configured: quotaConfigured, quotaMode } = require("./ai-quota");
const { configuredNames,geminiCredentials } = require("./ai-settings");

function configurationStatus(env = process.env) {
  const providers = availableProviders(env), missing = [];
  if (!providers.length) missing.push("GROQ_API_KEY or Groq_Key (or an explicitly enabled Gemini provider)");
  if (!quotaConfigured(env)) {
    if (quotaMode(env) === "firestore") {
      if (!env.FIREBASE_SERVICE_ACCOUNT_JSON) missing.push("FIREBASE_SERVICE_ACCOUNT_JSON");
      if ((env.AI_RATE_LIMIT_SECRET || "").length < 32) missing.push("AI_RATE_LIMIT_SECRET");
      if (env.FIREBASE_SERVICE_ACCOUNT_JSON && !missing.includes("FIREBASE_SERVICE_ACCOUNT_JSON")) missing.push("Valid FIREBASE_SERVICE_ACCOUNT_JSON");
    } else missing.push("AI_QUOTA_MODE must be instance or firestore");
  }
  return { configured: missing.length === 0, providers, missing, settingNames: configuredNames(env), quotaMode: quotaMode(env),
    geminiKeySlots:geminiCredentials(env).map(key=>key.slot), independentGeminiQuotas:new Set(geminiCredentials(env).map(key=>key.quotaGroup)).size>1,
    geminiOptInRequired: Boolean(geminiCredentials(env).length && env.GEMINI_ENABLED !== "true") };
}
module.exports = { configurationStatus };
