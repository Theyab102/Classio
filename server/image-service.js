const { providerCredentials } = require("./ai-settings");
const { credentialRequest } = require("./credential-request");
const { inspectImage } = require("../src/shared/image-format");
class ImageServiceError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}
const dataImage = (value) =>
  typeof value === "string" &&
  value.length <= 4200000 &&
  /^data:image\/(png|jpeg|webp);base64,[a-z0-9+/=]+$/i.test(value);
function imageCapabilities(env = process.env) {
  return {
    imageEditing: false,
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
