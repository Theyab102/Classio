import { useEffect, useRef, useState } from "react";
import { downloadStudyFile, TTS_SPACE_URL } from "./studyTools";

import { generatePodcastAudio, singleVoiceDialogue } from "./podcastAudio";

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
    pending.current = controller;
    setLoading(true);
    setStatus("Preparing audio. A sleeping voice server can take a moment to start.");
    try {
      const result = await generatePodcastAudio({language:'en',voices:['aria'],speed:1,dialogue:singleVoiceDialogue(script)},{signal:controller.signal});
      const audio=result.blob;
      if (controller.signal.aborted) return;
      downloadStudyFile(audio, audio.type.includes("mp4") ? "classio-podcast.m4a" : "classio-podcast.wav");
      setStatus("Audio downloaded.");
    } catch (error) {
      if (!controller.signal.aborted) setStatus(error.message);
    } finally {
      pending.current = null;
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  return (
    <div className="study-tool-actions">
      <button type="button" onClick={() => downloadStudyFile(new Blob([script], { type: "text/plain" }), "classio-podcast.txt")}>Download script</button>
      {TTS_SPACE_URL && <button type="button" disabled={loading} onClick={downloadAudio}>Download audio (English voice)</button>}
      {TTS_SPACE_URL && <p>Downloading audio sends this script to the Classio Piper voice service.</p>}
      {status && <p role="status">{status}</p>}
    </div>
  );
}
