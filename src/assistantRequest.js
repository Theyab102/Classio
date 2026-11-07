export function assistantHistory(messages, system) {
  const recent = messages.filter(message => !message.failed && ["user", "assistant"].includes(message.role)).slice(-30);
  const lastImage = recent.map(message => Boolean(message.image)).lastIndexOf(true);
  let remaining = Math.max(1000, 115000 - system.length);
  const bounded = [];
  for (let index = recent.length - 1; index >= 0; index--) {
    const message = recent[index], text = String(message.content || "").slice(0, 10000);
    if (text.length > remaining && bounded.length) break;
    const content = text.slice(0, remaining);
    remaining -= content.length;
    bounded.unshift({ role: message.role, content: message.image && index === lastImage ? [
      { type: "image_url", image_url: { url: message.image } }, { type: "text", text: content },
    ] : content });
  }
  return [{ role: "system", content: system }, ...bounded];
}
