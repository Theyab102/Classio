const { publicTranscript } = require("../server/video-transcript");
const { reserveRequest } = require("../server/ai-quota");
module.exports = async function transcript(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "GET") { res.setHeader("Allow", "GET"); return res.status(405).json({ error: "Use GET." }); }
  if (req.headers["sec-fetch-site"] && req.headers["sec-fetch-site"] !== "same-origin") return res.status(403).json({ error: "Open Classio to read a transcript." });
  const id = req.query?.v;
  if (typeof id !== "string" || !/^[A-Za-z0-9_-]{11}$/.test(id)) return res.status(400).json({ error: "Enter a valid public YouTube video link." });
  try {
    const limit = await reserveRequest(req, "transcript");
    if (limit.unavailable) return res.status(503).json({ error: "Transcript access is temporarily unavailable. You can paste the transcript manually." });
    if (limit.retryAfter) { res.setHeader("Retry-After", String(limit.retryAfter)); return res.status(429).json({ error: "Transcript requests are busy. Try again shortly or paste the transcript." }); }
    return res.status(200).json({ success: true, ...await publicTranscript(id) });
  } catch { return res.status(422).json({ success: false, error: "A public transcript could not be retrieved. Paste the video transcript to make source-based notes.", manualAvailable: true }); }
};
