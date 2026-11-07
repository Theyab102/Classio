const { configured, reserveRequest } = require("../server/ai-quota");
const { parseAudioRequest, parseStudyRequest } = require("../server/study-request");

const { generateStudy } = require("../server/ai-providers");
const { configurationStatus } = require("../server/ai-configuration");

async function requestQuota(req, action) {
  let timer;
  try {
    return await Promise.race([reserveRequest(req, action), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error("QUOTA_TIMEOUT")), 8000);
    })]);
  } finally { clearTimeout(timer); }
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
  if (site && site !== "same-origin") return res.status(403).json({ error: "Open Classio to use the study tools." });
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers["content-type"] || "")) {
    return res.status(415).json({ error: "Send a JSON study request." });
  }

  const audio = req.body?.kind === "audio";
  const input = audio ? parseAudioRequest(req.body) : parseStudyRequest(req.body);
  if (!input) return res.status(400).json({ error: "This request is invalid or too large. Check the file or shorten the input." });
  const report = configurationStatus();
  if (!configured() || !report.configured) {
    console.error("study_configuration", { missing: report.missing });
    return res.status(503).json({ error: "AI is unavailable because the site’s server settings are incomplete. Your writing is still saved in this session.", code: "AI_NOT_CONFIGURED" });
  }

  try {
    const quota = await requestQuota(req, audio ? "audio" : "study");
    if (quota.unavailable) return res.status(503).json({ error: "AI study tools are temporarily unavailable." });
    if (quota.retryAfter) {
      res.setHeader("Retry-After", String(quota.retryAfter));
      return res.status(429).json({ error: "The free AI request limit was reached. Try again later." });
    }
    const result = await generateStudy(input, { report: details => console.warn("study_provider", details) });
    return res.status(200).json(result);
  } catch (error) {
    const code = error.code || "QUOTA_UNAVAILABLE";
    console.error("study_request", { code });
    const messages = {
      CONTENT_BLOCKED: "The AI service could not answer this request. Rephrase it or use different study material.",
      CREDENTIALS_REJECTED: "The AI provider rejected its server credentials. The site owner needs to update the provider secret.",
      MODEL_UNAVAILABLE: "No supported text model is available for this provider. Check its model access or select an available model on the server.",
      VISION_MODEL_UNAVAILABLE: "Image analysis is unavailable for the configured provider. You can still ask a text question.",
      MODEL_CATALOG_UNAVAILABLE: "The AI provider could not list available models. Try again shortly.",
      OUTPUT_LIMIT: "The answer was too long to finish. Ask for a shorter response or a smaller course.",
    };
    const status = error.status || 503;
    if (status === 429) res.setHeader("Retry-After", "30");
    return res.status(status).json({ error: messages[code] || (status === 429
      ? "The free AI services are busy. Try again shortly."
      : "The AI service could not complete this request. Your work is still open; try again shortly."), code });
  }
};
