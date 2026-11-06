const { availableProviders } = require("./ai-providers");

function configurationStatus(env = process.env) {
  const providers = availableProviders(env);
  const missing = [];
  if (!providers.length) {
    if (env.GEMINI_ENABLED === "true") {
      if (!env.GEMINI_API_KEY) missing.push("GEMINI_API_KEY");
      if (!/^[\w.-]+$/.test(env.GEMINI_MODEL || "")) missing.push("GEMINI_MODEL");
    } else missing.push("GROQ_API_KEY");
  }
  let accountValid = false;
  try {
    const account = JSON.parse(env.FIREBASE_SERVICE_ACCOUNT_JSON || "");
    accountValid = Boolean(account && account.project_id && account.client_email && account.private_key);
  } catch {}
  if (!accountValid) missing.push("FIREBASE_SERVICE_ACCOUNT_JSON");
  if ((env.AI_RATE_LIMIT_SECRET || "").length < 32) missing.push("AI_RATE_LIMIT_SECRET");
  return { configured: missing.length === 0, providers, missing };
}

module.exports = { configurationStatus };
