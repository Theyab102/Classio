import { useEffect, useRef, useState } from "react";
import { downloadStudyFile, TTS_SPACE_URL } from "./studyTools";

export default function PodcastDownload({ script }) {
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const pending = useRef(null);

  useEffect(() => {
    return () => pending.current?.abort();
  }, []);

  async function downloadAudio() {
    if (pending.current) return;
    const controller = new AbortController();
    const timeout = AbortSignal.timeout(90000);
    pending.current = controller;
    setLoading(true);
    setStatus("Preparing audio. A sleeping voice server can take a moment to start.");
    try {
      const response = await fetch(`${TTS_SPACE_URL}/generate-podcast`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        signal: AbortSignal.any([controller.signal, timeout]),
        body: JSON.stringify({ text: script, voice: "aria", speed: 1 }),
      });
      if (!response.ok || !response.headers.get("content-type")?.startsWith("audio/")) {
        throw new Error("Audio download is temporarily unavailable. The script is still available.");
      }
      const audio = await response.blob();
      if (controller.signal.aborted) return;
      downloadStudyFile(audio, "classio-podcast.wav");
      setStatus("Audio downloaded.");
    } catch (error) {
      if (!controller.signal.aborted) setStatus(timeout.aborted
        ? "The voice server took too long. Try again or download the script."
        : error.message);
    } finally {
      pending.current = null;
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  return (
    <div className="study-tool-actions">
      <button type="button" onClick={() => downloadStudyFile(new Blob([script], { type: "text/plain" }), "classio-podcast.txt")}>Download script</button>
      {TTS_SPACE_URL && <button type="button" disabled={loading} onClick={downloadAudio}>Download WAV (English voice)</button>}
      {TTS_SPACE_URL && <p>Downloading audio sends this script to the Classio Piper voice service.</p>}
      {status && <p role="status">{status}</p>}
    </div>
  );
}
