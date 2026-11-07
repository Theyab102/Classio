export async function sendStudyRequest(payload, signal) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (signal?.aborted) cancel();
  signal?.addEventListener("abort", cancel, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 55000);
  try {
    const response = await fetch("/api/study", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || typeof data.text !== "string" || !data.text.trim()) {
      throw new Error(data.error || "The AI service could not finish this request. Your work is still open.");
    }
    return data.text;
  } catch (error) {
    if (timedOut) throw new Error("The AI service took too long. Try a shorter request or try again shortly.");
    if (!signal?.aborted && error instanceof TypeError) throw new Error("Could not reach the study service. Check your connection and try again.");
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

export const MAX_AUDIO_BYTES = 2_800_000;

export function mergeSessionFolders(accountFolders, sessionFolders) {
  const folders = new Map(accountFolders.map(folder => [folder.id, folder]));
  for (const folder of sessionFolders) {
    const saved = folders.get(folder.id);
    const files = new Map((saved?.files || []).map(file => [file.id, file]));
    for (const file of folder.files) files.set(file.id, file);
    folders.set(folder.id, { ...(saved || folder), files: [...files.values()] });
  }
  return [...folders.values()];
}

const reviewIntervals = [1, 3, 7, 14, 30];

export function scheduleCardReview(previous, remembered, now = Date.now()) {
  if (!remembered) return { level: 0, due: now + 10 * 60 * 1000 };
  const level = Math.min((previous?.level || 0) + 1, reviewIntervals.length);
  return { level, due: now + reviewIntervals[level - 1] * 24 * 60 * 60 * 1000 };
}

export const TTS_SPACE_URL = (process.env.REACT_APP_CLASSIO_TTS_URL ?? "https://theyab102-classio-tts.hf.space").replace(/\/$/, "");

export function downloadStudyFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function makeStudyId() {
  return window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
