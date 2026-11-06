import { useEffect, useRef, useState } from "react";
import { sendStudyRequest } from "./studyTools";
import "./studyTools.css";

export function parseLessonPlan(raw) {
  const value = JSON.parse(raw.replace(/^\s*```(?:json)?\s*|\s*```\s*$/gi, ""));
  if (!Array.isArray(value) || value.length < 2 || value.length > 6) throw new Error("The AI returned an incomplete course. Try again.");
  const text = (input, max) => {
    if (typeof input !== "string" || !input.trim()) throw new Error("The AI returned an incomplete lesson. Try again.");
    return input.trim().slice(0, max);
  };
  return value.map(unit => {
    if (!unit || !Array.isArray(unit.options) || unit.options.length !== 3 ||
        !Number.isInteger(unit.correct) || unit.correct < 0 || unit.correct > 2) {
      throw new Error("A knowledge check was incomplete. Try generating the course again.");
    }
    return {
      title: text(unit.title, 100), lesson: text(unit.lesson, 4000), question: text(unit.question, 500),
      options: unit.options.map(option => text(option, 250)), correct: unit.correct, explanation: text(unit.explanation, 700),
    };
  });
}

export default function LearnMode({ file, getFileText, onUpdate, onUseAsNotes }) {
  const [topic, setTopic] = useState("");
  const [active, setActive] = useState(0);
  const [choice, setChoice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const request = useRef(null);
  const units = file.course?.units || [];
  const completed = file.course?.completed || [];
  const unit = units[active];

  useEffect(() => () => request.current?.abort(), []);

  async function generate(event) {
    event.preventDefault();
    if (loading) return;
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError("");
    try {
      const source = (file.notes || await getFileText() || "").slice(0, 15000);
      const subject = topic.trim();
      if (!source && !subject) throw new Error("Enter a topic or add study material first.");
      const raw = await sendStudyRequest({ messages: [
        { role: "system", content: "Build a short, accurate self-paced course. Output only a JSON array of 3 to 5 objects. Each object has title, lesson (plain text with short paragraphs and a worked example), question, options (exactly three strings), correct (zero-based index), and explanation. Use the learner's language. Use the source when supplied and label any supplemental explanations." },
        { role: "user", content: `Topic: ${subject || file.name}\n\nStudy material:\n${source}` },
      ], maxTokens: 4200 }, controller.signal);
      const course = { units: parseLessonPlan(raw), completed: [] };
      if (controller.signal.aborted) return;
      onUpdate({ ...file, course });
      setActive(0);
      setChoice(null);
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure.message);
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }

  function answer(index) {
    setChoice(index);
    if (index === unit.correct && !completed.includes(active)) {
      onUpdate({ ...file, course: { units, completed: [...completed, active] } });
    }
  }

  function select(index) {
    setActive(index);
    setChoice(null);
  }

  return (
    <section className="study-tool" aria-labelledby="learn-title">
      <h2 id="learn-title">Lessons</h2>
      <p>Turn your notes or a topic into short lessons with practice questions.</p>
      <form onSubmit={generate} className="study-tool-actions">
        <label htmlFor="course-topic">Topic</label>
        <input id="course-topic" placeholder="Optional if this file has study material" value={topic} onChange={event => setTopic(event.target.value)} />
        <button type="submit" disabled={loading}>{loading ? "Generating…" : units.length ? "Generate new lessons" : "Generate lessons"}</button>
      </form>
      {error && <p role="alert">{error}</p>}
      {unit && (
        <>
          <p role="status">{completed.length} of {units.length} lessons completed</p>
          {onUseAsNotes && <button type="button" onClick={() => onUseAsNotes(units.map(item => `## ${item.title}\n\n${item.lesson}`).join("\n\n"))}>Add lessons to notes</button>}
          <nav aria-label="Course lessons" className="study-tool-actions">
            {units.map((item, index) => <button key={index} type="button" aria-current={index === active ? "step" : undefined} onClick={() => select(index)}>{index + 1}. {item.title}{completed.includes(index) ? " (done)" : ""}</button>)}
          </nav>
          <article className="study-lesson">
            <h3>{unit.title}</h3>
            <p className="study-lesson-text">{unit.lesson}</p>
            <h4>{unit.question}</h4>
            <div className="study-tool-actions">
              {unit.options.map((option, index) => <button type="button" key={index} aria-pressed={choice === index} onClick={() => answer(index)}>{option}</button>)}
            </div>
            {choice !== null && <p role="status">{choice === unit.correct ? `Correct. ${unit.explanation}` : "Try another answer."}</p>}
            {completed.includes(active) && active < units.length - 1 && <button type="button" onClick={() => select(active + 1)}>Next lesson</button>}
          </article>
        </>
      )}
    </section>
  );
}
