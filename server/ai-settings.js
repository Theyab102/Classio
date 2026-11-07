// This module is server-only. Never import it from src/.
const { createHash } = require("node:crypto");

function providerKey(provider, env = process.env) {
  const names = provider === "groq" ? ["GROQ_API_KEY", "Groq_Key"] : ["GEMINI_API_KEY", "Gemini_Key"];
  for (const name of names) {
    const value = env[name]?.trim();
    if (value && !/^(?:your[_ -]|replace[_ -]|example|placeholder)/i.test(value) && !value.includes("...")) return value;
  }
  return "";
}

function configuredNames(env = process.env) {
  return ["GROQ_API_KEY", "Groq_Key", "GEMINI_API_KEY", "Gemini_Key"].filter(name => Boolean(env[name]?.trim()));
}

function credentialId(provider, env) {
  return provider + ":" + createHash("sha256").update(providerKey(provider, env)).digest("hex");
}

module.exports = { providerKey, configuredNames, credentialId };
