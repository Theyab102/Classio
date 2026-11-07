import { useEffect, useRef, useState } from "react";
import { downloadStudyFile, makeStudyId, sendStudyRequest, TTS_SPACE_URL } from "./studyTools";
import { parseEpisode, synthesizeEpisode, spokenTurn } from "./podcastAudio";
import { useLearningSession, useStudyActivity } from "./LearningSession";
import FriendlyAvatar from "./FriendlyAvatar";
import { DEFAULT_CHARACTER } from "./avatarOptions";
import WorkspaceArt from "./WorkspaceArt";
import UiIcon from "./UiIcon";
import StudioDialog from "./StudioDialog";
import PodcastConversation from "./PodcastConversation";

const timeLabel = seconds => `${Math.floor((seconds || 0) / 60)}:${String(Math.floor((seconds || 0) % 60)).padStart(2, "0")}`;
const legacyTurns = script => (script.match(/[^.!?]+[.!?]*/g) || [script]).reduce((turns, sentence) => { const last = turns.at(-1); if (last && last.text.length + sentence.length < 900) last.text += sentence; else turns.push({ speaker: "A", text: sentence.trim(), section: "Saved episode" }); return turns; }, []);
const defaultHosts = { A: { ...DEFAULT_CHARACTER, name: "Mira", hairStyle: 6, top: "#528ca6" }, B: { ...DEFAULT_CHARACTER, name: "Jules", skin: "#bc8056", hair: "#28292b", hairStyle: 3, top: "#b9885e", glasses: 1 } };
export default function PodcastStudio({ file, onUpdate, getFileText, audioStore }) {
  const episodes = file.podcastEpisodes || (file.podcastScript ? [{ id: "legacy", title: file.name, turns: legacyTurns(file.podcastScript), position: 0, createdAt: 0 }] : []);
  const [selected, setSelected] = useState(file.podcastActiveId || episodes[0]?.id || ""), [length, setLength] = useState("short"), [sourceVersion, setSourceVersion] = useState("current");
  const episode = episodes.find(item => item.id === selected) || episodes[0];
  const [phase, setPhase] = useState("idle"), [progress, setProgress] = useState(""), [error, setError] = useState(""), [audioUrl, setAudioUrl] = useState(""), [blob, setBlob] = useState(null);
  const [position, setPosition] = useState(episode?.position || 0), [duration, setDuration] = useState(0), [playing, setPlaying] = useState(false), [speed, setSpeed] = useState(1), [volume, setVolume] = useState(.85), [view, setView] = useState("sections"), [remove, setRemove] = useState(false);
  const [talking, setTalking] = useState(false);
  const [voiceList, setVoiceList] = useState([]), [voices, setVoices] = useState({ A: "aria", B: "echo" });
  const audio = useRef(null), pending = useRef(null), latest = useRef({ file, onUpdate }), lastSaved = useRef(0), objectUrl = useRef(""); latest.current = { file, onUpdate };
  const { ask } = useLearningSession();
  const activeTurn = episode?.turns?.findIndex(turn => typeof turn.start === "number" && position >= turn.start && position < turn.end) ?? -1;
  useStudyActivity(file.id, "Podcast tutor", episode ? `Episode: ${episode.title}\nCurrent section: ${episode.turns[activeTurn]?.section || "Not playing"}\n${episode.turns[activeTurn]?.text || episode.turns.map(turn => turn.text).join("\n").slice(0, 5000)}` : "The learner is preparing a podcast.");
  useEffect(() => {
    if (!TTS_SPACE_URL) return;
    const controller = new AbortController();
    fetch(`${TTS_SPACE_URL}/voices`, { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(12000)]) }).then(response => response.ok ? response.json() : null).then(data => {
      if (!data || controller.signal.aborted) return;
      const items = Array.isArray(data) ? data.map(voice => typeof voice === "string" ? { id: voice, label: voice } : voice) : Object.entries(data).map(([id, item]) => ({ id, ...(typeof item === "object" ? item : { label: id }) }));
      const valid = items.filter(item => typeof item.id === "string");
      setVoiceList(valid);
      if (valid.length) setVoices({ A: valid.find(item => item.gender === "female")?.id || valid[0].id, B: valid.find(item => item.gender === "male")?.id || valid[1]?.id || valid[0].id });
    }).catch(() => {});
    return () => controller.abort();
  }, []);
  useEffect(() => () => { pending.current?.abort(); if (objectUrl.current) URL.revokeObjectURL(objectUrl.current); }, []);
  useEffect(() => {
    let live = true;
    audio.current?.pause(); setPlaying(false); setDuration(0); setPosition(episode?.position || 0); setAudioUrl(""); setBlob(null);
    if (objectUrl.current) { URL.revokeObjectURL(objectUrl.current); objectUrl.current = ""; }
    if (!episode?.audioKey) return;
    audioStore?.get(episode.audioKey).then(value => { if (!live) return; if (value instanceof Blob) { const url = URL.createObjectURL(value); objectUrl.current = url; setBlob(value); setAudioUrl(url); } else setError("This episode’s audio is stored on the device where it was created. The transcript is available here; you can prepare audio again."); }).catch(() => { if (live) setError("The saved audio could not be opened. You can prepare it again from this transcript."); });
    return () => { live = false; };
  // Audio is reloaded only when selecting an episode, not on each playback-position save.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episode?.id, episode?.audioKey, episode?.audioRevision]);
  useEffect(() => { if (audio.current) { audio.current.playbackRate = speed; audio.current.volume = volume; } }, [speed, volume, audioUrl]);
  function updateEpisodes(next, active = selected) { const data = latest.current; data.onUpdate({ ...data.file, podcastEpisodes: next, podcastActiveId: active }); }
  function savePosition(value, force = false) {
    if (!episode) return;
    if (!force && Math.abs(value - lastSaved.current) < 5) return;
    lastSaved.current = value;
    const items = latest.current.file.podcastEpisodes || episodes;
    updateEpisodes(items.map(item => item.id === episode.id ? { ...item, position: value } : item));
  }
  async function prepareAudio(target, controller) {
    setPhase("audio");
    const result = await synthesizeEpisode(target, voices, controller.signal, (part, total) => setProgress(`Preparing audio ${part} of ${total}`));
    if (controller.signal.aborted) return;
    const key = `podcast:${file.id}:${target.id}`;
    await audioStore.save(key, result.blob);
    if (controller.signal.aborted) return;
    const next = { ...target, turns: result.turns, duration: result.duration, audioKey: key, position: 0, voices: { ...voices }, audioVersion: 2, audioRevision: Date.now() };
    const items = latest.current.file.podcastEpisodes || episodes;
    updateEpisodes([next, ...items.filter(item => item.id !== target.id)], next.id); setSelected(next.id); setPosition(0); setError("");
  }
  async function generate() {
    if (pending.current) return;
    const controller = new AbortController(); pending.current = controller; setError(""); setPhase("script"); setProgress("Writing your episode"); audio.current?.pause();
    try {
      const source = (sourceVersion === "current" ? file.notes || await getFileText?.() : file.namedNotes?.[Number(sourceVersion)]?.text) || "";
      if (!source.trim()) throw new Error("Write notes or add readable source material before creating a podcast.");
      const raw = await sendStudyRequest({ messages: [{ role: "system", content: `Write a natural, accurate two-host educational podcast grounded in the supplied study material. Return only JSON with title and turns. Each turn has speaker (A or B), section (short descriptive heading), text (plain spoken words). Hosts are Mira and Jules. Put speaker identity only in the speaker field. NEVER start text with Mira:, Jules:, Host A:, a speaker label or a self-introduction. They speak naturally, without announcing their names. Use ${length === "short" ? "6 turns and 900 to 1800 characters total" : "8 to 10 turns and 3000 to 5500 characters total"}. Both hosts must contribute; use conversational questions, explanations and worked examples. Avoid repetitive introductions, invented facts, markdown and stage directions. Speak symbols and equations naturally. Use the source language.` }, { role: "user", content: `${file.name}\n\n${source.slice(0, 15000)}` }], maxTokens: length === "short" ? 2000 : 4500 }, controller.signal);
      if (controller.signal.aborted) return;
      const parsed = parseEpisode(raw), target = { ...parsed, id: makeStudyId(), createdAt: Date.now(), position: 0 };
      const items = latest.current.file.podcastEpisodes || episodes;
      updateEpisodes([target, ...items], target.id); setSelected(target.id);
      await prepareAudio(target, controller);
    } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
    finally { if (pending.current === controller) { if (!controller.signal.aborted) { setPhase("idle"); setProgress(""); } pending.current = null; } }
  }
  async function retryAudio() {
    if (pending.current || !episode) return;
    const controller = new AbortController(); pending.current = controller; setError("");
    try { await prepareAudio(episode, controller); } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
    finally { if (pending.current === controller) { if (!controller.signal.aborted) { setPhase("idle"); setProgress(""); } pending.current = null; } }
  }
  async function play() {
    if (!audio.current || !audioUrl) return;
    try { if (playing) audio.current.pause(); else await audio.current.play(); }
    catch { setError("Audio playback could not start. Try Play again or download the audio."); }
  }
  function seek(value) { if (!audio.current || !duration) return; const target = Math.max(0, Math.min(duration, value)); audio.current.currentTime = target; setPosition(target); savePosition(target, true); }
  const sections = episode?.turns?.reduce((items, turn, index) => { const last = items.at(-1); if (last?.name === turn.section) last.text += " " + spokenTurn(turn.text); else items.push({ name: turn.section, start: turn.start, text: spokenTurn(turn.text), index }); return items; }, []) || [];
  const busy = phase !== "idle";
  return <section className="podcast-studio"><header className="tool-heading"><div><p className="eyebrow">LISTEN TO LEARN</p><h2>Podcasts</h2><p>A conversation built from your own study material.</p></div>{!!episodes.length && <label className="episode-picker"><span className="sr-only">Episode history</span><select aria-label="Episode history" value={episode?.id || ""} disabled={busy} onChange={event => { savePosition(position, true); setSelected(event.target.value); setError(""); }}>{episodes.map(item => <option key={item.id} value={item.id}>{item.title}{item.createdAt ? ` · ${new Date(item.createdAt).toLocaleDateString()}` : ""}</option>)}</select></label>}</header>
    <div className="podcast-generator"><label>Length<select aria-label="Podcast length" value={length} disabled={busy} onChange={event => setLength(event.target.value)}><option value="short">Quick review</option><option value="long">Deep dive</option></select></label>
      {!!file.namedNotes?.length && <label>Material<select value={sourceVersion} onChange={event => setSourceVersion(event.target.value)}><option value="current">Current notes / source</option>{file.namedNotes.map((note, i) => <option key={i} value={i}>{note.name}</option>)}</select></label>}
      <button className="quiet-button is-primary" disabled={busy} onClick={generate}><UiIcon name="plus" />Create podcast</button>{busy && <button className="quiet-button" onClick={() => { pending.current?.abort(); pending.current = null; setPhase("idle"); setProgress(""); }}>Cancel generation</button>}
    </div>
    <details className="podcast-voice-options"><summary>Playback voices</summary><div>{["A", "B"].map(speaker => <label key={speaker}>{defaultHosts[speaker].name}<select value={voices[speaker]} onChange={event => setVoices(current => ({ ...current, [speaker]: event.target.value }))}>{(voiceList.length ? voiceList : [{ id: "aria", label: "Aria" }, { id: "echo", label: "Echo" }]).map(voice => <option key={voice.id} value={voice.id}>{voice.label || voice.id}</option>)}</select></label>)}</div><p>Voice changes apply the next time audio is prepared.{voiceList.length === 1 ? " This service currently offers one voice for both hosts." : ""}</p></details>
    {error && <p className="tool-error" role="alert">{error}</p>}{busy && <div className="podcast-preparing" role="status"><div className="mini-spinner" /><span>{progress}</span></div>}
    {episode ? <div className="podcast-layout"><article className="podcast-player"><div className="podcast-cover"><WorkspaceArt kind="audio" /><span>CLASSIO AUDIO</span></div><p className="eyebrow">YOUR STUDY EPISODE</p><h3>{episode.title}</h3><div className="podcast-hosts">{[...new Set(episode.turns.map(turn => turn.speaker))].map(speaker => <div key={speaker} className={playing && episode.turns[activeTurn]?.speaker === speaker ? "is-speaking" : ""}><FriendlyAvatar character={defaultHosts[speaker]} size={58} /><span>{defaultHosts[speaker].name}</span></div>)}</div>
      {audioUrl ? <><audio key={episode.id} ref={audio} src={audioUrl} preload="metadata" onLoadedMetadata={event => { setDuration(Number.isFinite(event.target.duration) ? event.target.duration : 0); const saved = episode.position || 0; if (saved < event.target.duration) event.target.currentTime = saved; }} onTimeUpdate={event => { setPosition(event.target.currentTime); savePosition(event.target.currentTime); }} onPlay={() => setPlaying(true)} onPause={() => { setPlaying(false); savePosition(audio.current?.currentTime || 0, true); }} onEnded={() => { setPlaying(false); savePosition(0, true); }} onError={() => setError("This audio file could not be played. Prepare the audio again or download it.")} />
        <label className="podcast-seek"><span className="sr-only">Playback position</span><input type="range" aria-label="Podcast playback position" min={0} max={duration || 1} step="0.1" value={Math.min(position, duration || 1)} disabled={!duration} onChange={event => seek(Number(event.target.value))} /><span>{timeLabel(position)}</span><span>{timeLabel(duration)}</span></label>
        <div className="podcast-controls"><button className="quiet-button" aria-label="Back 15 seconds" onClick={() => seek(position - 15)}><UiIcon name="rewind" /></button><button className="podcast-play" aria-label={playing ? "Pause podcast" : "Play podcast"} onClick={play}><UiIcon name={playing ? "pause" : "play"} size={27} /></button><button className="quiet-button" aria-label="Forward 15 seconds" onClick={() => seek(position + 15)}><UiIcon name="forward" /></button><select aria-label="Playback speed" value={speed} onChange={event => setSpeed(Number(event.target.value))}>{[.75, 1, 1.25, 1.5, 2].map(value => <option key={value} value={value}>{value}×</option>)}</select></div>
        <label className="podcast-volume"><UiIcon name="volume" size={18} /><span className="sr-only">Volume</span><input type="range" aria-label="Podcast volume" min="0" max="1" step="0.05" value={volume} onChange={event => setVolume(Number(event.target.value))} /></label></>
        : <div className="podcast-audio-missing"><p>{busy ? "Your transcript is ready. Audio is being prepared." : "The transcript is ready. Prepare real audio to listen."}</p><button className="quiet-button is-primary" disabled={busy} onClick={retryAudio}>Prepare audio</button></div>}
      {episode.turns.some(turn => spokenTurn(turn.text) !== turn.text) && <div className="podcast-audio-repair"><p>This saved audio may include spoken host labels.</p><button className="quiet-button" disabled={busy} onClick={retryAudio}>Rebuild audio without speaker labels</button></div>}
      <div className="podcast-secondary-actions"><button className="text-button" onClick={() => downloadStudyFile(new Blob([episode.turns.map(turn => `${defaultHosts[turn.speaker].name}: ${spokenTurn(turn.text)}`).join("\n\n")], { type: "text/plain" }), "classio-podcast.txt")}><UiIcon name="download" size={16} />Transcript</button>{blob && <button className="text-button" onClick={() => downloadStudyFile(blob, "classio-podcast.wav")}>Download audio</button>}<button className="text-button" onClick={() => setRemove(true)}>Delete episode</button></div>
      <button className="quiet-button podcast-ask" onClick={() => { audio.current?.pause(); setTalking(true); }}><UiIcon name="mic" />Ask the hosts aloud</button><button className="text-button" onClick={() => { audio.current?.pause(); ask(`Explain the ${episode.turns[activeTurn]?.section || episode.title} section of this podcast.`); }}>Open text tutor</button>
    </article><div className="podcast-transcript"><nav className="workspace-tabs" aria-label="Podcast transcript view"><button aria-pressed={view === "sections"} onClick={() => setView("sections")}>Sections</button><button aria-pressed={view === "transcript"} onClick={() => setView("transcript")}>Full transcript</button></nav>
      <div className="podcast-transcript-scroll">{view === "sections" ? sections.map((section, i) => <button className="podcast-section" key={i} disabled={!audioUrl || typeof section.start !== "number"} onClick={() => seek(section.start)} aria-current={activeTurn >= section.index && activeTurn < (sections[i + 1]?.index ?? episode.turns.length) ? "true" : undefined}><span><strong>{section.name}</strong>{typeof section.start === "number" && <small>{timeLabel(section.start)}</small>}</span><p>{section.text}</p></button>)
        : episode.turns.map((turn, i) => <button className={`podcast-turn speaker-${turn.speaker}`} key={i} disabled={!audioUrl || typeof turn.start !== "number"} onClick={() => seek(turn.start)} aria-current={i === activeTurn ? "true" : undefined}><strong>{defaultHosts[turn.speaker].name}</strong><span>{spokenTurn(turn.text)}</span></button>)}</div>
    </div></div> : !busy && <div className="tool-empty"><WorkspaceArt kind="audio" /><h3>Give your notes a voice</h3><p>Create a two-host episode from the current material. Real audio, a readable transcript and your playback position stay together.</p></div>}
    {talking && episode && <PodcastConversation episode={episode} section={episode.turns[Math.max(0, activeTurn)]} voice={episode.voices?.[episode.turns[Math.max(0, activeTurn)]?.speaker || "A"] || voices.A} onClose={() => setTalking(false)} onResume={() => { setTalking(false); audio.current?.play().catch(() => setError("Tap Play to resume the episode.")); }} onSave={item => { const items = latest.current.file.podcastEpisodes || episodes; updateEpisodes(items.map(current => current.id === episode.id ? { ...current, voiceQuestions: [...(current.voiceQuestions || []), { ...item, position }].slice(-20) } : current)); }} />}
    {remove && <StudioDialog title="Delete podcast?" onClose={() => setRemove(false)}><p>This removes the selected episode and its saved audio from this workspace.</p><div className="dialog-actions"><button className="quiet-button" onClick={() => setRemove(false)}>Cancel</button><button className="quiet-button is-danger" onClick={() => { audio.current?.pause(); if (episode.audioKey) audioStore.remove?.(episode.audioKey); const next = episodes.filter(item => item.id !== episode.id); updateEpisodes(next, next[0]?.id || ""); setSelected(next[0]?.id || ""); setRemove(false); }}>Delete episode</button></div></StudioDialog>}
  </section>;
}
