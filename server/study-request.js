const MAX_TEXT_LENGTH = 120_000;
const MAX_IMAGE_LENGTH = 3_800_000;
const supportedRoles = new Set(["system", "user", "assistant"]);
const imageData = /^data:image\/(?:jpeg|png|webp);base64,[A-Za-z0-9+/]+={0,2}$/;

function parseStudyRequest(body) {
  if (!body || !Array.isArray(body.messages) || !body.messages.length || body.messages.length > 64) return null;
  if (!Number.isInteger(body.maxTokens) || body.maxTokens < 1 || body.maxTokens > 8_000) return null;

  let textLength = 0;
  let imageCount = 0;
  const messages = [];
  for (const message of body.messages) {
    if (!message || !supportedRoles.has(message.role)) return null;
    const { role, content } = message;
    if (typeof content === "string") {
      textLength += content.length;
      messages.push({ role, content });
      continue;
    }

    if (role !== "user" || !Array.isArray(content) || content.length !== 2 || imageCount) return null;
    const [image, text] = content;
    const url = image?.image_url?.url;
    if (image?.type !== "image_url" || typeof url !== "string" ||
        url.length > MAX_IMAGE_LENGTH || !imageData.test(url) ||
        text?.type !== "text" || typeof text.text !== "string") return null;
    imageCount++;
    textLength += text.text.length;
    messages.push({ role, content: [{ type: "image_url", image_url: { url } }, { type: "text", text: text.text }] });
  }
  if (textLength > MAX_TEXT_LENGTH || !messages.some(message => message.role === "user")) return null;
  const tasks=new Set(["classification","style","notes","questions","flashcards","conversation","presentation","avatar-fit"]);
  return { messages, maxTokens: body.maxTokens, vision: imageCount === 1, ...(tasks.has(body.task)?{task:body.task}:{}) };
}

function parseAudioRequest(body) {
  const { audio, filename } = body || {};
  if (typeof audio !== "string" || !audio.length || audio.length > 3_800_000 ||
      audio.length % 4 !== 0 || !/^[A-Za-z0-9+/]+={0,2}$/.test(audio) ||
      typeof filename !== "string" || !/^[^/\\\r\n]{1,128}\.(flac|mp3|mp4|mpeg|mpga|m4a|ogg|wav|webm)$/i.test(filename)) return null;
  return { audio, filename };
}

module.exports = { parseAudioRequest, parseStudyRequest };
