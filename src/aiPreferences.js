export function aiMode() {
  try {
    return localStorage.getItem("classio_ai_mode") || "auto";
  } catch {
    return "auto";
  }
}
export function setAiMode(value) {
  try {
    localStorage.setItem("classio_ai_mode", value);
  } catch {}
  window.dispatchEvent(new Event("classio-ai-mode"));
}
