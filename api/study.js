const { authorize } = require("../server/ai-auth");
const { accessFailure } = require("../server/access-errors");
const { configured, reserveRequest } = require("../server/ai-quota");
const {
  parseAudioRequest,
  parseStudyRequest,
} = require("../server/study-request");

const { generateStudy } = require("../server/ai-providers");
const { configurationStatus } = require("../server/ai-configuration");
const {
  safetyResponse,
  SAFETY_INSTRUCTION,
} = require("../server/study-safety");

async function requestQuota(req, action) {
  let timer;
  try {
    return await Promise.race([
      reserveRequest(req, action),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("QUOTA_TIMEOUT")), 8000);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

module.exports = async function study(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST." });
  }

  // All supported browser calls are same-origin JSON. Requests that lack a
  // browser fetch context are still subject to the same server quotas.
  const site = req.headers["sec-fetch-site"];
  if (site && site !== "same-origin")
    return res
      .status(403)
      .json({ error: "Open Classio to use the study tools." });
  if (
    !/^application\/json(?:\s*;|$)/i.test(req.headers["content-type"] || "")
  ) {
    return res.status(415).json({ error: "Send a JSON study request." });
  }

  try {
    req.aiIdentity = await authorize(req);
  } catch (error) {
    const failure = accessFailure(error);
    console.warn("study_access", {
      code: failure.body.code,
      ...(error.reason ? { reason: error.reason } : {}),
    });
    return res.status(failure.status).json(failure.body);
  }

  const audio = req.body?.kind === "audio";
  const input = audio
    ? parseAudioRequest(req.body)
    : parseStudyRequest(req.body);
  if (!input)
    return res.status(400).json({
      error:
        "This request is invalid or too large. Check the file or shorten the input.",
    });
  const supportive = safetyResponse(input);
  if (supportive) return res.status(200).json({ text: supportive });
  if (input.messages)
    input.messages = [
      { role: "system", content: SAFETY_INSTRUCTION },
      ...input.messages,
    ];
  const report = configurationStatus();
  if (!configured() || !report.configured) {
    console.error("study_configuration", {
      code: "AI_NOT_CONFIGURED",
      missing: report.missing,
      disabledProviders: Object.entries(report.providerConfiguration)
        .filter(([, value]) => !value.enabled)
        .map(([provider, value]) => ({ provider, reason: value.reason })),
    });
    return res.status(503).json({
      error: "AI study tools are currently unavailable. Your writing is kept.",
      code: "AI_NOT_CONFIGURED",
    });
  }

  let lease;
  try {
    const quota = await requestQuota(req, audio ? "audio" : "study");
    if (quota.unavailable)
      return res.status(503).json({
        error: "AI study tools are temporarily unavailable.",
        code: "QUOTA_UNAVAILABLE",
      });
    if (quota.retryAfter) {
      res.setHeader("Retry-After", String(quota.retryAfter));
      return res.status(429).json({
        error: `The free AI request limit was reached. Try again in ${quota.retryAfter} seconds.`,
        code: "CLIENT_RATE_LIMIT",
      });
    }
    lease = quota.release;
    const result = await generateStudy(input, {
      identity: req.aiIdentity.sub,
      report: (details) => console.warn("study_provider", details),
    });
    return res.status(200).json(result);
  } catch (error) {
    const code = error.code || "QUOTA_UNAVAILABLE";
    console.error("study_request", { code });
    const messages = {
      CONTENT_BLOCKED:
        "The AI service could not answer this request. Rephrase it or use different study material.",
      CREDENTIALS_REJECTED:
        "AI is temporarily unavailable. Your saved progress is kept.",
      MODEL_UNAVAILABLE:
        "This AI model is unavailable. Try Auto or another available provider.",
      VISION_MODEL_UNAVAILABLE:
        "Image analysis is unavailable for the configured provider. You can still ask a text question.",
      MODEL_CATALOG_UNAVAILABLE:
        "The AI provider could not list available models. Try again shortly.",
      OUTPUT_LIMIT:
        "The AI response ended before it was complete. Your current work is kept. Try generating this section again.",
      REQUEST_TOO_LARGE:
        "This request exceeds the provider’s token allowance. Use a shorter section or ask for fewer notes or slides. Small questions are still available.",
      NETWORK:
        "The AI provider did not respond in time. Your work is saved; try this question again.",
      EMPTY_RESPONSE:
        "The AI provider returned an empty answer. Try this question again.",
      INVALID_RESPONSE:
        "The AI provider returned an unreadable answer. Try this question again.",
    };
    const status = error.status || 503;
    if (status === 429)
      res.setHeader("Retry-After", String(error.retryAfter || 30));
    return res.status(status).json({
      error:
        messages[code] ||
        (status === 429
          ? "The free AI services are busy. Try again shortly."
          : "The AI service could not complete this request. Your work is still open; try again shortly."),
      code,
    });
  } finally {
    await lease?.();
  }
};
