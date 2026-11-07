function playerResponse(html) {
  const marker = /(?:var\s+)?ytInitialPlayerResponse\s*=\s*/g;
  const match = marker.exec(html);
  if (!match) return null;
  const start = html.indexOf("{", marker.lastIndex);
  let depth = 0, quote = false, escape = false;
  for (let index = start; index >= 0 && index < html.length; index++) {
    const letter = html[index];
    if (quote) { if (escape) escape = false; else if (letter === "\\") escape = true; else if (letter === '"') quote = false; continue; }
    if (letter === '"') quote = true;
    else if (letter === "{") depth++;
    else if (letter === "}" && --depth === 0) { try { return JSON.parse(html.slice(start, index + 1)); } catch { return null; } }
  }
  return null;
}
const entities = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
function decodeText(text) {
  return text.replace(/&(#x[\da-f]+|#\d+|amp|lt|gt|quot|apos);/gi, (_, value) => {
    if (value[0] !== "#") return entities[value.toLowerCase()] || "";
    const number = value[1].toLowerCase() === "x" ? parseInt(value.slice(2), 16) : Number(value.slice(1));
    return number > 0 && number <= 0x10ffff ? String.fromCodePoint(number) : "";
  }).replace(/<[^>]*>/g, "").replace(/\s+/g, " ").trim();
}
function captionText(raw) {
  try { const json = JSON.parse(raw); return (json.events || []).flatMap(event => event.segs || []).map(segment => segment.utf8 || "").join(" ").replace(/\s+/g, " ").trim(); } catch {}
  return [...raw.matchAll(/<(?:text|p)\b[^>]*>([\s\S]*?)<\/(?:text|p)>/g)].map(match => decodeText(match[1])).filter(Boolean).join(" ");
}
async function publicTranscript(videoId, fetcher = global.fetch) {
  const response = await fetcher(`https://www.youtube.com/watch?v=${videoId}`, { headers: { "Accept-Language": "en-US,en;q=0.8" }, signal: AbortSignal.timeout(8000) });
  if (!response.ok) throw new Error("TRANSCRIPT_UNAVAILABLE");
  const player = playerResponse((await response.text()).slice(0, 4000000));
  const tracks = player?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
  if (!Array.isArray(tracks) || !tracks.length) throw new Error("TRANSCRIPT_UNAVAILABLE");
  const track = tracks.find(item => item.languageCode === "en") || tracks[0];
  const url = new URL(track.baseUrl);
  if (url.protocol !== "https:" || !(url.hostname === "youtube.com" || url.hostname.endsWith(".youtube.com")) || url.username || url.password) throw new Error("TRANSCRIPT_UNAVAILABLE");
  url.searchParams.set("fmt", "json3");
  const captions = await fetcher(url.href, { signal: AbortSignal.timeout(8000) });
  if (!captions.ok) throw new Error("TRANSCRIPT_UNAVAILABLE");
  const transcript = captionText((await captions.text()).slice(0, 3000000)).slice(0, 80000);
  if (!transcript) throw new Error("TRANSCRIPT_UNAVAILABLE");
  return { transcript, title: typeof player.videoDetails?.title === "string" ? player.videoDetails.title.slice(0, 200) : "YouTube video" };
}
module.exports = { publicTranscript, playerResponse, captionText };
