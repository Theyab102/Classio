const { configurationStatus } = require("../server/ai-configuration");

module.exports = function status(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (!["GET", "HEAD"].includes(req.method)) {
    res.setHeader("Allow", "GET, HEAD");
    return res.status(405).json({ error: "Use GET." });
  }
  // Names and presence only. Never return keys, account JSON, or provider data.
  const report = configurationStatus();
  const body = { aiConfigured: report.configured, providers: report.providers, missing: report.missing,
    checked: "configuration only; live provider access is not tested" };
  if (req.method === "HEAD") return res.status(report.configured ? 200 : 503).end();
  return res.status(report.configured ? 200 : 503).json(body);
};
