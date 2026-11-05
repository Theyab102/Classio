const { configured, reserveRequest } = require("../server/ai-quota");
const { parseAudioRequest, parseStudyRequest } = require("../server/study-request");

const GROQ_URL = "https://api.groq.com/openai/v1";

async function studyResponse(input) {
  if (input.audio) {
    const form = new FormData();
    form.set("model", "whisper-large-v3-turbo");
    form.set("file", new Blob([Buffer.from(input.audio, "base64")]), input.filename);
    return fetch(`${GROQ_URL}/audio/transcriptions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
      body: form,
      signal: AbortSignal.timeout(30_000),
    });
  }

  const model = input.vision
    ? process.env.GROQ_VISION_MODEL || "qwen/qwen3.8-27b"
    : process.env.GROQ_TEXT_MODEL || "openai/gpt-oss-120b";
  return fetch(`${GROQ_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
    body: JSON.stringify({ model, messages: input.messages, max_completion_tokens: input.maxTokens }),
    signal: AbortSignal.timeout(25_000),
  });
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
  if (!configured()) return res.status(503).json({ error: "AI study tools are unavailable." });

  try {
    const quota = await reserveRequest(req, audio ? "audio" : "study");
    if (quota.unavailable) return res.status(503).json({ error: "AI study tools are temporarily unavailable." });
    if (quota.retryAfter) {
      res.setHeader("Retry-After", String(quota.retryAfter));
      return res.status(429).json({ error: "The free AI request limit was reached. Try again later." });
    }
    const upstream = await studyResponse(input);
    if (upstream.status === 429) return res.status(429).json({ error: "The free AI service is busy. Try again later." });
    if (!upstream.ok) return res.status(502).json({ error: audio ? "The audio could not be transcribed." : "The AI service could not complete this request." });
    const data = await upstream.json();
    const text = audio ? data.text : data.choices?.[0]?.message?.content;
    if (typeof text !== "string" || !text.trim()) return res.status(502).json({ error: "The AI service returned an empty response." });
    return res.status(200).json({ text });
  } catch {
    return res.status(503).json({ error: "AI study tools are temporarily unavailable." });
  }
};
