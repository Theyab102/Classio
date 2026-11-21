const { providerCredentials } = require("./ai-settings");
const { credentialRequest } = require("./credential-request");
const { inspectImage } = require("../src/shared/image-format");
const { providerPolicy } = require("./ai-policy");
const IMAGE_MODEL = "@cf/black-forest-labs/flux-1-schnell";
class ImageServiceError extends Error {
  constructor(status, message, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}
const dataImage = (value) =>
  typeof value === "string" &&
  value.length <= 4200000 &&
  /^data:image\/(png|jpeg|webp);base64,[a-z0-9+/=]+$/i.test(value);
function imageCapabilities(env = process.env) {
  return {
    imageEditing: false,
    imageGeneration:
      env.AI_ENABLE_IMAGE_GENERATION === "true" &&
      env.AI_IMAGE_MODEL_APPROVED === "true" &&
      providerPolicy("cloudflare", env).enabled,
    backgroundRemoval:
      providerCredentials("removebg", env).length > 0 &&
      (env.AI_ALLOW_PAID === "true" ||
        env.REMOVE_BG_FREE_TIER_CONFIRMED === "true"),
  };
}
async function imageOperation(
  input,
  { env = process.env, fetcher = global.fetch } = {},
) {
  const capability = imageCapabilities(env);
  if (input.action === "generate") {
    if (!capability.imageGeneration)
      throw new ImageServiceError(
        503,
        "Image generation is unavailable. Upload an image or search for a figure instead.",
        "IMAGE_NOT_CONFIGURED",
      );
    if (
      typeof input.prompt !== "string" ||
      input.prompt.trim().length < 5 ||
      input.prompt.length > 1700
    )
      throw new ImageServiceError(
        400,
        "Describe the illustration in 5–1,700 characters.",
        "IMAGE_INVALID_REQUEST",
      );
    // Image generation has its own model, budget and quota. Text chat, vision
    // analysis and avatar sketch refinement must never reach this endpoint.
    const prompt =
      "Create an original classroom illustration appropriate for students aged 13+. No sexual content, graphic violence or impersonation. Do not claim factual accuracy for invented diagrams. Requested illustration: " +
      input.prompt.trim();
    const signal = AbortSignal.timeout(35000);
    for (const credential of providerCredentials("cloudflare", env)) {
      if (!/^[a-f\d]{32}$/i.test(credential.accountId || "")) continue;
      let response, data;
      try {
        response = await fetcher(
          `https://api.cloudflare.com/client/v4/accounts/${credential.accountId}/ai/run/${IMAGE_MODEL}`,
          {
            method: "POST",
            redirect: "error",
            signal,
            headers: {
              Authorization: `Bearer ${credential.key}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt, steps: 4 }),
          },
        );
        data = await response.json();
      } catch {
        throw new ImageServiceError(
          503,
          "The image service could not respond. Your original image is kept.",
          "IMAGE_UNAVAILABLE",
        );
      }
      if (
        data?.result?.blocked ||
        (data?.errors || []).some((error) =>
          /content_policy|moderation|safety/i.test(
            String(error.code) + " " + String(error.message),
          ),
        )
      )
        throw new ImageServiceError(
          422,
          "The image service could not accept this request. Try a different illustration.",
          "IMAGE_CONTENT_BLOCKED",
        );
      if (
        [5016, 5018, 3023, 3041, 5035].includes(Number(data?.errors?.[0]?.code))
      )
        throw new ImageServiceError(
          503,
          "The image service needs its account access checked. Your original image is kept.",
          "IMAGE_UNAVAILABLE",
        );
      if ([401, 403].includes(response.status)) continue;
      if (response.status === 429) {
        const error = new ImageServiceError(
          429,
          "The free image service is busy. Try again shortly.",
          "IMAGE_RATE_LIMIT",
        );
        error.retryAfter = Math.min(
          3600,
          Math.max(1, Number(response.headers?.get?.("retry-after")) || 60),
        );
        throw error;
      }
      if (!response.ok || data?.success === false)
        throw new ImageServiceError(
          503,
          "The image service is unavailable. Your original image is kept.",
          "IMAGE_UNAVAILABLE",
        );
      const encoded = data?.result?.image;
      if (
        typeof encoded !== "string" ||
        encoded.length > 4000000 ||
        !/^[a-z0-9+/]+={0,2}$/i.test(encoded)
      )
        throw new ImageServiceError(
          502,
          "The image service returned an unusable illustration.",
          "IMAGE_INVALID_RESPONSE",
        );
      const bytes = Buffer.from(encoded, "base64"),
        format = inspectImage(bytes);
      if (
        !format ||
        !["image/png", "image/jpeg", "image/webp"].includes(format.mime) ||
        format.animated ||
        !bytes.length ||
        bytes.length > 3000000
      )
        throw new ImageServiceError(
          502,
          "The image service returned an unusable illustration.",
          "IMAGE_INVALID_RESPONSE",
        );
      return { image: `data:${format.mime};base64,${encoded}` };
    }
    throw new ImageServiceError(
      503,
      "The image service needs its server configuration checked. Your original image is kept.",
      "IMAGE_UNAVAILABLE",
    );
  }
  if (input.action === "remove-background") {
    if (!capability.backgroundRemoval)
      throw new ImageServiceError(
        503,
        "Background removal is unavailable. Your original image is kept.",
      );
    if (!dataImage(input.image))
      throw new ImageServiceError(
        400,
        "Choose a PNG, JPEG or WebP image under 3 MB. Convert static GIFs before uploading; animated images are unsupported.",
      );
    const bytesIn = Buffer.from(input.image.split(",")[1], "base64"),
      format = inspectImage(bytesIn),
      claimed = input.image.slice(5, input.image.indexOf(";"));
    if (
      !format ||
      format.mime !== claimed ||
      format.animated ||
      bytesIn.length > 3000000
    )
      throw new ImageServiceError(
        400,
        "Choose a valid, still PNG, JPEG or WebP image under 3 MB. The original is kept.",
      );
    const form = new FormData();
    form.set("image_file_b64", input.image.split(",")[1]);
    form.set("size", "auto");
    form.set("format", "png");
    let response;
    try {
      response = await credentialRequest(
        "removebg",
        (credential) => ({
          url: "https://api.remove.bg/v1.0/removebg",
          method: "POST",
          headers: { "X-Api-Key": credential.key },
          body: form,
        }),
        { env, fetcher, signal: AbortSignal.timeout(35000) },
      );
    } catch {
      throw new ImageServiceError(
        504,
        "Background removal timed out or could not connect. Your original is kept; try again.",
      );
    }
    if (!response.ok) {
      const messages = {
        401: "The background-removal service rejected its server credential. The site owner needs to update it.",
        403: "The background-removal credential does not have access. Contact the site owner.",
        402: "The background-removal account has no credits available. Your original is kept.",
        429: "The background-removal service reached its request limit. Try again shortly.",
        400: "The service could not identify a foreground subject. Try a clear object photo.",
        413: "The image exceeds the provider’s size limit. Upload a smaller image.",
        415: "The provider does not support this image format. Use PNG, JPEG or WebP.",
      };
      throw new ImageServiceError(
        response.status === 429
          ? 429
          : response.status === 400 ||
              response.status === 413 ||
              response.status === 415
            ? 422
            : 502,
        messages[response.status] ||
          "The background-removal service could not process this image. Your original is kept; try again later.",
      );
    }
    const bytes = Buffer.from(await response.arrayBuffer());
    if (
      bytes.length > 3000000 ||
      bytes.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a"
    )
      throw new ImageServiceError(
        502,
        "The service returned an unsupported cutout.",
      );
    return { image: "data:image/png;base64," + bytes.toString("base64") };
  }
  throw new ImageServiceError(
    503,
    "AI sketch refinement is unavailable for this student workspace. Drawing, manual adjustments and private drafts remain available.",
  );
}
module.exports = {
  imageCapabilities,
  imageOperation,
  ImageServiceError,
  dataImage,
};
