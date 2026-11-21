const { authorize, AccessError } = require("../server/ai-auth");
const { accessFailure } = require("../server/access-errors");
const { accessDiagnostic } = require("../server/access-diagnostic");
const { configured, reserveRequest } = require("../server/ai-quota");
const {
  imageCapabilities,
  imageOperation,
  ImageServiceError,
} = require("../server/image-service");
const {
  searchProviders,
  searchImages,
  trackImageDownload,
} = require("../server/image-search");
module.exports = async function images(req, res) {
  res.setHeader("Cache-Control", "no-store");
  if (req.method === "GET") {
    const capabilities = imageCapabilities();
    const ready =
      configured() && (process.env.CLASSIO_SESSION_SECRET || "").length >= 32;
    return res.status(200).json({
      ...capabilities,
      imageGeneration: ready && capabilities.imageGeneration,
      backgroundRemoval: ready && capabilities.backgroundRemoval,
      searchProviders: searchProviders(),
    });
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ error: "Use POST." });
  }
  if (
    (req.headers["sec-fetch-site"] &&
      req.headers["sec-fetch-site"] !== "same-origin") ||
    !/^application\/json(?:\s*;|$)/i.test(req.headers["content-type"] || "")
  )
    return res.status(403).json({ error: "Open Classio to edit images." });
  let release;
  try {
    req.aiIdentity = await authorize(req);
    const quota = await reserveRequest(
      req,
      req.body?.action === "search"
        ? "image-search"
        : req.body?.action === "track-download"
          ? "image-attribution"
          : "image",
    );
    if (quota.unavailable)
      return res.status(503).json({ error: "Image requests are unavailable." });
    if (quota.retryAfter) {
      res.setHeader("Retry-After", String(quota.retryAfter));
      return res.status(429).json({
        error: "The image request limit was reached. Try again later.",
      });
    }
    release = quota.release;
    if (req.body?.action === "search")
      return res.status(200).json(await searchImages(req.body.query));
    if (req.body?.action === "track-download")
      return res
        .status(200)
        .json({ tracked: await trackImageDownload(req.body.id) });
    return res.status(200).json(await imageOperation(req.body || {}));
  } catch (error) {
    if (error instanceof AccessError) {
      const failure = accessFailure(error);
      console.warn("images_access", accessDiagnostic(error, failure.body.code));
      return res.status(failure.status).json(failure.body);
    }
    if (error instanceof ImageServiceError) {
      if (error.retryAfter)
        res.setHeader("Retry-After", String(error.retryAfter));
      return res.status(error.status).json({
        error: error.message,
        ...(error.code ? { code: error.code } : {}),
      });
    }
    return res.status(503).json({
      error: "The image service is unavailable. Your original image is kept.",
    });
  } finally {
    await release?.();
  }
};
