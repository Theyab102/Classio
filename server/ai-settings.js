// Server-only registry. Exact deployed names take precedence; legacy aliases stay supported.
const { createHash } = require("node:crypto");
const hash = (value) => createHash("sha256").update(value).digest("hex");
const registry = {
  groq: {
    primary: ["Groq_Key", "GROQ_API_KEY"],
    backup: ["Groq_Key_packup", "GROQ_API_KEY_packup", "GROQ_API_KEY_2"],
  },
  gemini: {
    primary: ["Gemini_Key", "GEMINI_API_KEY"],
    backup: [
      "Gemini_Key_packup",
      "GEMINI_API_KEY_2",
      "Gemini_Key_2",
      "Gemini_Key2",
      "GEMINI_API_KEY2",
    ],
  },
  openrouter: {
    primary: ["OPENROUTER_API_KEY"],
    backup: ["OPENROUTER_API_KEY_packup"],
  },
  cloudflare: {
    primary: ["CLOUDFLARE_API_TOKEN"],
    backup: ["CLOUDFLARE_API_TOKEN_packup"],
  },
  ollama: {
    primary: ["OLLAMA_API_TOKEN"],
    backup: ["OLLAMA_API_TOKEN_packup"],
  },
  removebg: {
    primary: ["REMOVE_BG_API_KEY"],
    backup: ["REMOVE_BG_API_KEY_packup"],
  },
  pexels: { primary: ["PEXELS_API_KEY"], backup: ["PEXELS_API_KEY_packup"] },
  unsplash: {
    primary: ["UNSPLASH_ACCESS_KEY"],
    backup: ["UNSPLASH_ACCESS_KEY_packup"],
  },
  googleimages: {
    primary: ["GOOGLE_IMAGE_SEARCH_KEY"],
    backup: ["GOOGLE_IMAGE_SEARCH_KEY_packup"],
  },
};
function first(names, env) {
  for (const name of names) {
    const value = env[name]?.trim();
    if (
      value &&
      !/^(?:your[_ -]|replace[_ -]|example|placeholder)/i.test(value) &&
      !value.includes("...")
    )
      return { key: value, name };
  }
  return null;
}
function providerCredentials(provider, env = process.env) {
  const spec = registry[provider];
  if (!spec) return [];
  const seen = new Set();
  const rows = [first(spec.primary, env), first(spec.backup, env)];
  return rows.flatMap((value, index) => {
    if (!value || seen.has(value.key)) return [];
    seen.add(value.key);
    const accountId =
      provider === "cloudflare"
        ? (index === 1
            ? env.CLOUDFLARE_ACCOUNT_ID_packup || env.CLOUDFLARE_ACCOUNT_ID
            : env.CLOUDFLARE_ACCOUNT_ID
          )?.trim()
        : undefined;
    return [
      {
        ...value,
        provider,
        slot: index + 1,
        role: index ? "backup" : "primary",
        id: hash(value.key),
        accountId,
        quotaGroup: hash(
          provider === "cloudflare" ? accountId || "missing" : provider,
        ),
      },
    ];
  });
}
function providerKey(provider, env = process.env) {
  return env._activeCredential?.provider === provider
    ? env._activeCredential.key
    : providerCredentials(provider, env)[0]?.key || "";
}
function credentialId(provider, env = process.env) {
  return provider + ":" + hash(providerKey(provider, env));
}
function credentialEnvironment(env, credential) {
  return {
    ...env,
    _activeCredential: credential,
    _geminiQuotaGroup: credential.quotaGroup,
    _geminiSlot: credential.slot,
  };
}
function configuredNames(env = process.env) {
  return Object.values(registry)
    .flatMap((spec) => [...spec.primary, ...spec.backup])
    .filter((name) => !!env[name]?.trim());
}
function credentialStatus(provider, env = process.env) {
  const keys = providerCredentials(provider, env);
  return {
    primary: keys.some((key) => key.role === "primary"),
    backup: keys.some((key) => key.role === "backup"),
  };
}
module.exports = {
  registry,
  providerCredentials,
  providerKey,
  credentialId,
  credentialEnvironment,
  configuredNames,
  credentialStatus,
  geminiCredentials: (env) => providerCredentials("gemini", env),
};
