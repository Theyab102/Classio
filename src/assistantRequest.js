export function isSimpleGreeting(text) {
  return /^(?:(?:hi|hey|hay|hello|hiya|bonjour|salut|hola|مرحبا|مرحباً|اهلا|أهلاً|السلام عليكم|你好)(?:\s+classio)?)[\s!.?،؟]*$/i.test(String(text).trim());
}

export function materialExcerpt(text, question, limit = 6500) {
  const source = String(text || ""); if (source.length <= limit) return source;
  const terms = new Set(String(question).toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || []);
  ["explain", "what", "please", "give", "main", "ideas", "this", "about", "example"].forEach(word => terms.delete(word));
  const blocks = source.split(/\n\s*\n/).map((content, index) => ({ content, index, score: [...terms].reduce((score, term) => score + (content.toLowerCase().includes(term) ? 1 : 0), 0) }));
  const chosen = new Map(); let length = 0;
  const add = block => { if (!block || chosen.has(block.index) || length >= limit) return; const content = block.content.slice(0, limit - length); chosen.set(block.index, content); length += content.length + 2; };
  add(blocks[0]);
  const ranked = blocks.filter(block => block.score).sort((a, b) => b.score - a.score || a.index - b.index);
  if (ranked.length) ranked.forEach(add);
  else for (let i = 0; i < Math.min(blocks.length, 10); i++) add(blocks[Math.floor(i * blocks.length / Math.min(blocks.length, 10))]);
  return [...chosen].sort((a, b) => a[0] - b[0]).map(([, content]) => content).join("\n\n").slice(0, limit);
}

export function assistantHistory(messages, system, budget = 14000) {
  system = String(system).slice(0, Math.max(1000, budget - 2000));
  const recent = messages.filter(message => !message.failed && ["user", "assistant"].includes(message.role)).slice(-30);
  const lastImage = recent.map(message => Boolean(message.image)).lastIndexOf(true);
  let remaining = Math.max(1000, budget - system.length);
  const bounded = [];
  for (let index = recent.length - 1; index >= 0; index--) {
    const message = recent[index], text = String(message.content || "").slice(0, 10000);
    const hasUser = bounded.some(item => item.role === "user");
    if (text.length > remaining && bounded.length && (hasUser || message.role !== "user")) break;
    const content = text.slice(0, message.role === "assistant" && !hasUser ? Math.floor(remaining / 2) : remaining);
    remaining -= content.length;
    bounded.unshift({ role: message.role, content: message.image && index === lastImage ? [
      { type: "image_url", image_url: { url: message.image } }, { type: "text", text: content },
    ] : content });
  }
  return [{ role: "system", content: system }, ...bounded];
}
