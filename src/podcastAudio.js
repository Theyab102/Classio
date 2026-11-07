import {spokenMathText} from "./mathSpeech";
import { TTS_SPACE_URL } from "./studyTools";

export function spokenTurn(text) {
  // Speaker names belong in transcript labels, not in the spoken sentence.
  return String(text || "").replace(/^(?:\s*\*{0,2}(?:Mira|Jules|Host\s*[AB12]|Speaker\s*[AB12]|[AB])\*{0,2}\s*[:：]\s*)+/i, "").trim();
}

export async function synthesizeSpeech(text, voice, signal) {
  if (!TTS_SPACE_URL) throw new Error("Audio generation is not connected.");
  const response = await fetch(`${TTS_SPACE_URL}/generate-podcast`, { method: "POST", headers: { "Content-Type": "application/json" }, cache: "no-store", signal: AbortSignal.any([signal, AbortSignal.timeout(90000)]), body: JSON.stringify({ text: spokenMathText(spokenTurn(text)), voice, speed: 1 }) });
  if (!response.ok || !response.headers.get("content-type")?.startsWith("audio/")) throw new Error("The voice service could not create audio. Your transcript is still available.");
  const blob = await response.blob();
  if (blob.size > 15000000) throw new Error("This audio section is too large. Try a shorter answer.");
  return blob;
}

export function parseEpisode(raw) {
  const value = JSON.parse(raw.trim().replace(/^```(?:json)?\s*|\s*```$/gi, ""));
  if (!value || typeof value.title !== "string" || !value.title.trim() || !Array.isArray(value.turns) || value.turns.length < 4 || value.turns.length > 12) throw new Error("The AI returned an incomplete podcast. Try again.");
  let length = 0;
  const turns = value.turns.map(turn => {
    if (!turn || !["A", "B"].includes(turn.speaker) || typeof turn.text !== "string" || !turn.text.trim() || typeof turn.section !== "string" || !turn.section.trim()) throw new Error("A podcast section was incomplete. Try again.");
    length += turn.text.length;
    const text = spokenTurn(turn.text);
    if (!text) throw new Error("A podcast turn contains a label without an explanation. Try again.");
    return { speaker: turn.speaker, text: text.slice(0, 1800), section: turn.section.trim().slice(0, 100) };
  });
  if (length > 11000 || !turns.some(turn => turn.speaker === "A") || !turns.some(turn => turn.speaker === "B")) throw new Error("The podcast needs a shorter conversation with both hosts. Try again.");
  return { title: value.title.trim().slice(0, 150), turns };
}

export function pcmWave(chunks, sampleRate = 22050) {
  const size = chunks.reduce((total, chunk) => total + chunk.length, 0), data = new ArrayBuffer(44 + size * 2), view = new DataView(data);
  const word = (offset, value) => [...value].forEach((letter, i) => view.setUint8(offset + i, letter.charCodeAt(0)));
  word(0, "RIFF"); view.setUint32(4, 36 + size * 2, true); word(8, "WAVE"); word(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); word(36, "data"); view.setUint32(40, size * 2, true);
  let offset = 44;
  for (const chunk of chunks) for (const sample of chunk) { const bounded = Math.max(-1, Math.min(1, sample)); view.setInt16(offset, bounded * (bounded < 0 ? 32768 : 32767), true); offset += 2; }
  return new Blob([data], { type: "audio/wav" });
}

export async function synthesizeEpisode(episode, voices, signal, onProgress = () => {}) {
  if (!TTS_SPACE_URL) throw new Error("Audio generation is not connected. Your transcript is available below.");
  const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Offline) throw new Error("This browser cannot prepare podcast audio. Try a supported browser; your transcript stays available.");
  const context = new Offline(1, 1, 22050), chunks = [], turns = [];
  let frames = 0;
  for (let index = 0; index < episode.turns.length; index++) {
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    const turn = episode.turns[index]; onProgress(index + 1, episode.turns.length);
    const spoken = spokenTurn(turn.text);
    const bytes = await (await synthesizeSpeech(spoken, voices[turn.speaker], signal)).arrayBuffer();
    let buffer;
    try { buffer = await context.decodeAudioData(bytes); } catch { throw new Error("The voice service returned an unreadable audio file. Try preparing audio again."); }
    if (signal.aborted) throw new DOMException("Stopped", "AbortError");
    const samples = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) { const source = buffer.getChannelData(channel); for (let i = 0; i < samples.length; i++) samples[i] += source[i] / buffer.numberOfChannels; }
    turns.push({ ...turn, text: spoken, start: frames / 22050, end: (frames + samples.length) / 22050 });
    chunks.push(samples); frames += samples.length;
    if (frames > 22050 * 1200) throw new Error("The episode exceeds 20 minutes. Generate a shorter review.");
  }
  return { blob: pcmWave(chunks), turns, duration: frames / 22050 };
}
