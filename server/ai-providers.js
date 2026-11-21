const { createHash } = require("node:crypto");
const { providerCredentials } = require("./ai-settings");
const { eligibleProviders } = require("./ai-policy");
const adapters = require("./ai-adapters");
const { ProviderError } = require("./ai-errors");
const { workload, modality } = require("./ai-routing");
const health = new Map(),
  invalid = new Map(),
  inFlight = new Map(),
  cache = new Map();
function availableProviders(env = process.env) {
  const allowed = eligibleProviders(env),
    order = (env.AI_PROVIDER_ORDER || "groq,cloudflare,openrouter,ollama")
      .split(",")
      .map((value) => value.trim());
  return [...new Set([...order, ...allowed])].filter((provider) =>
    allowed.includes(provider),
  );
}
function providerOrder(input, env) {
  const enabled = availableProviders(env),
    suggested = input.vision
      ? ["groq", "openrouter", "ollama"]
      : ["groq", "cloudflare", "openrouter", "ollama"];
  const preferred =
    input.provider && input.provider !== "auto"
      ? input.provider
      : env.AI_ROUTING_MODE || "auto";
  return [
    ...new Set([
      preferred,
      ...(env.AI_PROVIDER_ORDER ? enabled : suggested),
      ...enabled,
    ]),
  ].filter(
    (provider) =>
      enabled.includes(provider) && (!input.audio || provider === "groq"),
  );
}
async function bounded(milliseconds, work) {
  const controller = new AbortController();
  let timer;
  try {
    return await Promise.race([
      work(controller.signal),
      new Promise(
        (_, reject) =>
          (timer = setTimeout(() => {
            controller.abort();
            reject(new ProviderError(503, "NETWORK", true));
          }, milliseconds)),
      ),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
function safeReport(report, value) {
  report({
    provider: value.provider,
    code: value.code,
    ...(value.role ? { credential: value.role } : {}),
    ...(value.workload ? { workload: value.workload } : {}),
  });
}
async function route(
  input,
  { env = process.env, fetcher = global.fetch, report = () => {} },
) {
  const providers = providerOrder(input, env);
  if (!providers.length) throw new ProviderError(503, "NOT_CONFIGURED");
  const deadline = Date.now() + 40000;
  let last;
  for (const provider of providers) {
    const credentials = providerCredentials(provider, env);
    const circuitKey = [
      provider,
      modality(input),
      ...credentials.map((key) => key.id + ":" + (key.accountId || "")),
    ].join(":");
    const circuit = health.get(circuitKey);
    if (circuit?.until > Date.now()) {
      last = new ProviderError(
        circuit.code === "PROVIDER_RATE_LIMIT" ? 429 : 503,
        circuit.code,
        true,
        Math.ceil((circuit.until - Date.now()) / 1000),
      );
      continue;
    }
    if (provider === "ollama" && !credentials.length)
      credentials.push({
        provider,
        role: "primary",
        slot: 1,
        key: "",
        id: "local",
      });
    for (const credential of credentials) {
      if (
        provider === "cloudflare" &&
        !/^[a-f\d]{32}$/i.test(credential.accountId || "")
      )
        continue;
      if (invalid.get(credential.id) > Date.now()) {
        last = new ProviderError(502, "CREDENTIALS_REJECTED", true);
        continue;
      }
      if (Date.now() >= deadline)
        throw last || new ProviderError(503, "NETWORK", true);
      try {
        const budget = Math.max(
          1,
          Math.min(
            providers.indexOf(provider) < providers.length - 1
              ? Math.min(
                  18000,
                  Math.max(1000, Number(env.AI_PRIMARY_TIMEOUT_MS) || 12000),
                )
              : 18000,
            deadline - Date.now(),
          ),
        );
        const result = await bounded(budget, async (signal) => {
          const context = { env, fetcher, signal },
            tried = [];
          let first;
          for (let attempt = 0; attempt < 2; attempt++) {
            let model;
            try {
              model = await adapters.choose(
                provider,
                input,
                credential,
                context,
                tried,
              );
            } catch (error) {
              throw first || error;
            }
            if (signal.aborted) throw new ProviderError(503, "NETWORK", true);
            tried.push(model.id);
            safeReport(report, {
              provider,
              role: credential.role,
              model: model.id,
              code: "ROUTE_SELECTED",
              workload: workload(input),
            });
            try {
              return {
                text: await adapters.generate(
                  provider,
                  input,
                  credential,
                  model,
                  context,
                ),
                provider,
              };
            } catch (error) {
              if (
                !["MODEL_UNAVAILABLE", "OUTPUT_LIMIT"].includes(error.code) ||
                attempt === 1
              )
                throw error;
              first = error;
              await new Promise((resolve) =>
                setTimeout(resolve, 150 * 2 ** attempt),
              );
            }
          }
        });
        health.delete(circuitKey);
        return result;
      } catch (error) {
        if (!(error instanceof ProviderError))
          error = new ProviderError(503, "PROVIDER_UNAVAILABLE", true);
        last = error;
        safeReport(report, {
          provider,
          role: credential.role,
          code: error.code,
          workload: workload(input),
        });
        if (!error.retryable) throw error;
        if (error.code === "CREDENTIALS_REJECTED") {
          invalid.set(credential.id, Date.now() + 300000);
          continue;
        }
        if (
          error.code === "PROVIDER_RATE_LIMIT" ||
          error.code === "NETWORK" ||
          /^HTTP_5|PROVIDER_UNAVAILABLE/.test(error.code)
        ) {
          const previous = health.get(circuitKey),
            failures = Math.min(5, (previous?.failures || 0) + 1),
            seconds =
              error.code === "PROVIDER_RATE_LIMIT"
                ? error.retryAfter || 60
                : Math.min(120, 5 * 2 ** failures);
          health.set(circuitKey, {
            provider,
            modality: modality(input),
            code: error.code,
            until: Date.now() + seconds * 1000,
            failures,
          });
        }
        // A second key is a credential backup, never extra shared quota/capacity.
        break;
      }
    }
  }
  throw last || new ProviderError(503, "MODEL_UNAVAILABLE", true);
}
async function generateStudy(input, options = {}) {
  const env = options.env || process.env,
    identity = options.identity || "server",
    fingerprint = JSON.stringify([
      identity,
      input,
      availableProviders(env),
      ...availableProviders(env).map((provider) =>
        providerCredentials(provider, env).map((key) => [
          key.id,
          key.accountId,
        ]),
      ),
      env.AI_ALLOW_PAID,
      env.AI_GROQ_MODEL,
      env.GROQ_TEXT_MODEL,
      env.GROQ_VISION_MODEL,
      env.AI_OPENROUTER_MODEL,
      env.AI_CLOUDFLARE_MODEL,
      env.OLLAMA_BASE_URL,
      env.OLLAMA_MODEL,
      env.AI_ROUTING_MODE,
      ...Object.keys(env)
        .filter((name) =>
          /^AI_(?:GROQ|CLOUDFLARE|OPENROUTER)_(?:FAST|STUDY|VISION|AUDIO)_MODEL$/.test(
            name,
          ),
        )
        .sort()
        .map((name) => [name, env[name]]),
    ]);
  const key = createHash("sha256").update(fingerprint).digest("hex"),
    saved = cache.get(key);
  if (saved?.until > Date.now()) return saved.value;
  if (inFlight.has(key)) return inFlight.get(key);
  if (inFlight.size >= 32) throw new ProviderError(503, "SERVER_BUSY", true);
  const work = route(input, options).then((value) => {
    if (["classification", "style"].includes(input.task)) {
      if (cache.size >= 100) cache.delete(cache.keys().next().value);
      cache.set(key, { value, until: Date.now() + 300000 });
    }
    return value;
  });
  inFlight.set(key, work);
  try {
    return await work;
  } finally {
    if (inFlight.get(key) === work) inFlight.delete(key);
  }
}
module.exports = {
  availableProviders,
  generateStudy,
  geminiPayload: adapters.geminiPayload,
  ProviderError,
  credentialHealth: (env = process.env) =>
    Object.fromEntries(
      ["groq", "gemini", "cloudflare", "openrouter", "ollama"].map(
        (provider) => [
          provider,
          Object.fromEntries(
            ["primary", "backup"].map((role) => {
              const credential = providerCredentials(provider, env).find(
                (key) => key.role === role,
              );
              return [
                role,
                !credential
                  ? "not-configured"
                  : invalid.get(credential.id) > Date.now()
                    ? "credential-invalid"
                    : "available",
              ];
            }),
          ),
        ],
      ),
    ),
  providerHealth: () => {
    const result = {};
    for (const value of health.values()) {
      if (value.until <= Date.now()) continue;
      const row = (result[value.provider] ||= {
        status: "available",
        capabilities: {},
      });
      row.capabilities[value.modality] = "temporarily-limited";
      if (value.modality === "text") row.status = "temporarily-limited";
    }
    return result;
  },
  clearProviderCaches: () => {
    health.clear();
    invalid.clear();
    inFlight.clear();
    cache.clear();
    adapters.clearCatalogs();
  },
};
