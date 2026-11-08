import {spokenMathText} from "./mathSpeech";
import { MATH_OUTPUT_GUIDE } from "./mathMarkdown";
import { useEffect, useRef, useState } from "react";
import { sendStudyRequest } from "./studyTools";
import { useStudyActivity } from "./LearningSession";
import StudyMarkdown from "./StudyMarkdown";
import WorkspaceArt from "./WorkspaceArt";
import UiIcon from "./UiIcon";
import "./studyTools.css";

export function parseLessonPlan(raw) {
  const value = JSON.parse(raw.replace(/^\s*```(?:json)?\s*|\s*```\s*$/gi, ""));
  if (!Array.isArray(value) || value.length < 2 || value.length > 6) throw new Error("The AI returned an incomplete course. Try again.");
  const text = (input, max) => { if (typeof input !== "string" || !input.trim()) throw new Error("The AI returned an incomplete lesson. Try again."); return input.trim().slice(0, max); };
  return value.map(unit => {
    if (!unit || !Array.isArray(unit.options) || unit.options.length !== 3 || !Number.isInteger(unit.correct) || unit.correct < 0 || unit.correct > 2) throw new Error("A knowledge check was incomplete. Try generating the course again.");
    return { title: text(unit.title, 100), lesson: text(unit.lesson, 5000), question: text(unit.question, 500), options: unit.options.map(option => text(option, 250)), correct: unit.correct, explanation: text(unit.explanation, 1000) };
  });
}
export default function LearnMode({ file, getFileText, onUpdate, onUseAsNotes }) {
  const [topic, setTopic] = useState("");
  const [active, setActive] = useState(() => Math.max(0, Math.min(file.course?.active || 0, (file.course?.units?.length || 1) - 1)));
  const [choice, setChoice] = useState(() => file.course?.answers?.[file.course?.active || 0] ?? null), [loading, setLoading] = useState(false), [error, setError] = useState(""), [speaking, setSpeaking] = useState(false);
  const request = useRef(null), latest = useRef({ file, onUpdate }); latest.current = { file, onUpdate };
  const units = file.course?.units || [], completed = file.course?.completed || [], unit = units[active];
  useStudyActivity(file.id, "Lesson tutor", unit ? `${unit.title}\n${unit.lesson}\nKnowledge check: ${unit.question}` : "The learner is preparing a course.", unit?.title);
  useEffect(() => () => { request.current?.abort(); window.speechSynthesis?.cancel(); }, []);
  const updateCourse = patch => { const current = latest.current; const course = { ...current.file.course, ...patch }; current.onUpdate({ ...current.file, course }); };
  async function generate(event) {
    event.preventDefault(); if (loading) return;
    if (units.length && !window.confirm("Generate a new course for this file? The current lesson plan will be replaced; your written notes stay saved.")) return;
    const controller = new AbortController(); request.current = controller; setLoading(true); setError("");
    try {
      const source = (file.notes || await getFileText?.() || "").slice(0, 15000);
      if (!source.trim() && !topic.trim()) throw new Error("Enter a topic or add study material first.");
      const raw = await sendStudyRequest({ messages: [
        { role: "system", content: `Build a short, accurate self-paced course. Output only a JSON array of 3 to 5 objects. Each object has title, lesson (Markdown with short titled sections, a worked example, and a short takeaway), question, options (exactly three strings), correct (zero-based index), and explanation. Use the learner's language. Stay grounded in the source, label supplemental explanations, and do not invent source content. ${MATH_OUTPUT_GUIDE}` },
        { role: "user", content: `Topic: ${topic.trim() || file.name}\n\nStudy material:\n${source}` },
      ], maxTokens: 5500 }, controller.signal);
      if (controller.signal.aborted) return;
      updateCourse({ units: parseLessonPlan(raw), completed: [], active: 0, answers: {} }); setActive(0); setChoice(null);
    } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
    finally { if (!controller.signal.aborted) setLoading(false); request.current = null; }
  }
  function select(index) {
    window.speechSynthesis?.cancel(); setSpeaking(false); setActive(index); setChoice(file.course?.answers?.[index] ?? null); updateCourse({ active: index });
  }
  function answer(index) {
    setChoice(index); updateCourse({ answers: { ...file.course?.answers, [active]: index }, completed: index === unit.correct && !completed.includes(active) ? [...completed, active] : completed });
  }
  function listen() {
    if (speaking) { window.speechSynthesis.cancel(); setSpeaking(false); return; }
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) { setError("Read aloud is not supported by this browser."); return; }
    const speech = new SpeechSynthesisUtterance(spokenMathText(`${unit.title}. ${unit.lesson}`));
    speech.lang = navigator.language; speech.rate = .95; speech.onend = () => setSpeaking(false); speech.onerror = () => { setSpeaking(false); setError("Read aloud stopped. You can continue reading the lesson."); };
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(speech); setSpeaking(true);
  }
  return <section className="lesson-studio" aria-labelledby="learn-title"><header className="tool-heading"><div><p className="eyebrow">STEP BY STEP</p><h2 id="learn-title">Guided lessons</h2><p>AI explanations with worked examples and helpful questions after each lesson, saved inside Notes.</p></div>
    {unit && <span className="tool-count">{completed.length} / {units.length} complete</span>}</header>
    <form onSubmit={generate} className={`course-generator ${unit ? "is-compact" : ""}`}><label htmlFor="course-topic">Topic<input id="course-topic" placeholder="Optional when this file has study material" value={topic} onChange={event => setTopic(event.target.value)} /></label><button className="quiet-button is-primary" disabled={loading}><UiIcon name="generate" />{loading ? "Generating lessons…" : unit ? "New course" : "Generate lessons"}</button></form>
    {error && <p role="alert" className="tool-error">{error}</p>}
    {!unit && !loading && <div className="tool-empty"><WorkspaceArt /><h3>Turn a chapter into a clear learning path</h3><p>Use the notes or source in this file, or enter a topic above. Your generated lessons and progress stay with this material.</p></div>}
    {loading && <div className="lesson-skeleton" role="status" aria-label="Generating lessons"><span /><span /><span /></div>}
    {unit && <><div className="course-progress"><div role="progressbar" aria-label="Course progress" aria-valuemin={0} aria-valuemax={units.length} aria-valuenow={completed.length}><span style={{ width: `${completed.length / units.length * 100}%` }} /></div><p role="status">{completed.length} of {units.length} lessons completed</p></div>
      <div className="lesson-layout"><nav className="lesson-outline" aria-label="Course lessons">{units.map((item, index) => <button key={index} onClick={() => select(index)} aria-current={active === index ? "step" : undefined}><span>{completed.includes(index) ? <UiIcon name="check" size={15} /> : String(index + 1).padStart(2, "0")}</span><strong><StudyMarkdown inline text={item.title}/></strong></button>)}</nav>
        <article className="lesson-paper"><div className="lesson-paper-heading"><p className="eyebrow">LESSON {active + 1} OF {units.length}</p>{window.speechSynthesis && <button className="quiet-button" onClick={listen} aria-pressed={speaking}><UiIcon name="volume" />{speaking ? "Stop reading" : "Listen"}</button>}</div><h3><StudyMarkdown inline text={unit.title}/></h3><StudyMarkdown text={unit.lesson} />
          <section className="lesson-check" aria-label="Knowledge check"><p className="eyebrow">CHECK YOUR UNDERSTANDING</p><StudyMarkdown className="lesson-check-question" text={unit.question} /><div className="answer-choices">{unit.options.map((option, index) => <button type="button" key={index} className={choice === index ? index === unit.correct ? "is-correct" : "is-incorrect" : ""} aria-pressed={choice === index} onClick={() => answer(index)}><span>{String.fromCharCode(65 + index)}</span><StudyMarkdown text={option} />{choice === index && <UiIcon name={index === unit.correct ? "check" : "close"} size={17} />}</button>)}</div>
          {choice !== null && <div className="answer-explanation" role="status"><strong>{choice === unit.correct ? "That’s right." : "Take another look."}</strong><StudyMarkdown text={unit.explanation} /></div>}</section>
          <footer className="lesson-navigation"><button className="quiet-button" disabled={active === 0} onClick={() => select(active - 1)}><UiIcon name="back" />Previous</button><span>{active + 1} / {units.length}</span>
            {active < units.length - 1 ? <button className="quiet-button is-primary" onClick={() => select(active + 1)}>Continue<UiIcon name="arrow" /></button> : <button className="quiet-button is-primary" disabled={choice !== unit.correct && !completed.includes(active)} onClick={() => { if (!completed.includes(active)) updateCourse({ completed: [...completed, active] }); }}><UiIcon name="check" />Finish lesson</button>}</footer>
        </article></div>{onUseAsNotes && <button className="text-button lesson-save-notes" onClick={() => onUseAsNotes(units.map(item => `## ${item.title}\n\n${item.lesson}`).join("\n\n"))}><UiIcon name="notes" />Add lessons to notes</button>}</>}
  </section>;
}
