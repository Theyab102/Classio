const { publicStatus } = require("../server/ai-configuration");
module.exports = function status(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (!["GET", "HEAD"].includes(req.method)) {
    res.setHeader("Allow", "GET, HEAD");
    return res.status(405).json({ error: "Use GET." });
  }
  if (req.method === "HEAD") return res.status(200).end();
  return res.status(200).json(publicStatus());
};
