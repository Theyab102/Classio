const { publicSource } = require("../server/public-source");
const { reserveRequest } = require("../server/ai-quota");
module.exports = async function source(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "POST") { res.setHeader("Allow", "POST"); return res.status(405).json({ error: "Use POST." }); }
  if (req.headers["sec-fetch-site"] && req.headers["sec-fetch-site"] !== "same-origin") return res.status(403).json({ error: "Open Classio to read a source." });
  if (!/^application\/json(?:\s*;|$)/i.test(req.headers["content-type"] || "") || typeof req.body?.url !== "string" || req.body.url.length > 2048) return res.status(400).json({ error: "Enter a public webpage URL." });
  let timer;
  try {
    const quota = await Promise.race([reserveRequest(req, "source"), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("QUOTA_TIMEOUT")), 8000); })]);
    clearTimeout(timer);
    if (quota.unavailable) return res.status(503).json({ error: "Source reading is unavailable. You can paste the page text." });
    if (quota.retryAfter) { res.setHeader("Retry-After", String(quota.retryAfter)); return res.status(429).json({ error: "Source requests are busy. Try again later." }); }
    return res.status(200).json(await publicSource(req.body.url));
  } catch { return res.status(422).json({ error: "This public page could not be read. It may block access or require sign-in. Paste its text to make notes.", manualAvailable: true }); }
  finally { clearTimeout(timer); }
};
