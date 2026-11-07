import ChoiceMenu from "./ChoiceMenu";
import { MATH_OUTPUT_GUIDE } from "./mathMarkdown";
import { useEffect, useRef, useState } from "react";
import { makeStudyId, sendStudyRequest } from "./studyTools";
import { useStudyActivity } from "./LearningSession";
import StudioDialog from "./StudioDialog";
import StudyMarkdown from "./StudyMarkdown";
import WorkspaceArt from "./WorkspaceArt";
import UiIcon from "./UiIcon";

export function parseQuiz(raw) {
  const value = JSON.parse(raw.trim().replace(/^```(?:json)?\s*|\s*```$/gi, ""));
  const rows = Array.isArray(value) ? value : value.questions;
  if (!Array.isArray(rows) || !rows.length || rows.length > 20) throw new Error("The AI returned an incomplete quiz. Try again.");
  return rows.map(row => {
    if (!row || typeof row.question !== "string" || !row.question.trim() || !Array.isArray(row.options) || row.options.length < 3 || row.options.length > 4 || row.options.some(text => typeof text !== "string" || !text.trim()) || !Number.isInteger(row.correct) || row.correct < 0 || row.correct >= row.options.length || typeof row.explanation !== "string" || !row.explanation.trim() || typeof row.hint !== "string" || !row.hint.trim()) throw new Error("A quiz question was incomplete. Try generating again.");
    return { question: row.question.trim().slice(0, 1500), options: row.options.map(text => text.trim().slice(0, 500)), correct: row.correct, hint: row.hint.slice(0, 800), explanation: row.explanation.slice(0, 1600) };
  });
}
export default function QuizStudio({ file, onUpdate, getFileText }) {
  const [settingsOpen, setSettingsOpen] = useState(false), [count, setCount] = useState(file.quiz?.settings?.count || 5), [difficulty, setDifficulty] = useState(file.quiz?.settings?.difficulty || "standard"), [topic, setTopic] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [hint, setHint] = useState(false), [review, setReview] = useState(false);
  const pending = useRef(null), latest = useRef({ file, onUpdate }); latest.current = { file, onUpdate };
  const quiz = file.quiz, questions = quiz?.questions || [], index = Math.min(quiz?.active || 0, Math.max(questions.length - 1, 0)), current = questions[index], answers = quiz?.answers || {}, checked = quiz?.checked || {};
  const correct = questions.filter((question, i) => checked[i] && answers[i] === question.correct).length;
  const done = Object.keys(checked).filter(key => checked[key]).length;
  useStudyActivity(file.id, "Quiz hints", current ? `${current.question}\nChoices:\n${current.options.join("\n")}${checked[index] ? `\nExplanation: ${current.explanation}` : "\nGive a hint first without revealing the answer."}` : "The learner is preparing a quiz.");
  useEffect(() => () => pending.current?.abort(), []);
  const update = patch => { const data = latest.current; data.onUpdate({ ...data.file, quiz: { ...data.file.quiz, ...patch } }); };
  async function generate(event) {
    event?.preventDefault(); if (busy) return;
    if (questions.length && !window.confirm("Generate a new quiz? The current attempt will be replaced; your materials stay saved.")) return;
    const controller = new AbortController(); pending.current = controller; setBusy(true); setError("");
    try {
      const source = (file.notes || await getFileText?.() || "").trim();
      if (!source && !topic.trim()) throw new Error("Add study material or enter a topic before generating a quiz.");
      const raw = await sendStudyRequest({ messages: [
        { role: "system", content: `Create ${count} accurate ${difficulty} multiple-choice study questions. Return only a JSON array of objects with question, options (four strings), correct (zero-based index), hint (helps reasoning without giving the answer), and explanation. Use the supplied source. Do not invent facts about unavailable sources. ${MATH_OUTPUT_GUIDE}` },
        { role: "user", content: `Topic: ${topic.trim() || file.name}\n\nSource:\n${source.slice(0, 15000)}` },
      ], maxTokens: Math.min(8000, count * 430 + 600) }, controller.signal);
      if (controller.signal.aborted) return;
      const next = parseQuiz(raw);
      if (next.length !== count) throw new Error(`The AI returned ${next.length} questions instead of ${count}. Try a smaller quiz.`);
      update({ id: makeStudyId(), questions: next, answers: {}, checked: {}, active: 0, completed: false, settings: { count, difficulty } }); setSettingsOpen(false); setReview(false); setHint(false);
    } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
    finally { if (!controller.signal.aborted) setBusy(false); pending.current = null; }
  }
  const go = next => { update({ active: next }); setHint(false); };
  const finished = quiz?.completed;
  return <section className="quiz-studio"><header className="tool-heading"><div><p className="eyebrow">ACTIVE RECALL</p><h2>Quizzes</h2><p>Check what you understand, then work through the explanations.</p></div><button className="quiet-button" onClick={() => setSettingsOpen(true)}><UiIcon name="settings" />Quiz settings</button></header>
    {error && <p className="tool-error" role="alert">{error}</p>}
    {!current ? <><form onSubmit={generate} className="course-generator"><label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with study material" /></label><button className="quiet-button is-primary" disabled={busy}><UiIcon name="generate" />{busy ? "Generating quiz…" : "Generate quiz"}</button></form>
      {!busy && <div className="tool-empty"><WorkspaceArt /><h3>Put your understanding into practice</h3><p>Generate questions from this material. Each question includes a hint and an explanation, and your answers stay with this file.</p></div>}</>
      : finished ? <div className="quiz-results"><span className="result-icon"><UiIcon name="check" size={32} /></span><p className="eyebrow">QUIZ COMPLETE</p><h3>{correct} of {questions.length} correct</h3><p>{done < questions.length ? `${questions.length - done} questions were not checked.` : "Use the explanations to revisit anything that felt uncertain."}</p><div className="studio-controls"><button className="quiet-button is-primary" onClick={() => setReview(value => !value)}>{review ? "Hide explanations" : "Review answers"}</button><button className="quiet-button" onClick={() => { update({ answers: {}, checked: {}, active: 0, completed: false }); setReview(false); }}>Try again</button></div>
        {review && <div className="quiz-answer-review">{questions.map((question, i) => <article key={i}><StudyMarkdown text={`${i + 1}. ${question.question}`} /><StudyMarkdown text={`Your answer: ${question.options[answers[i]] || "Not answered"}`} /><StudyMarkdown text={`Correct answer: ${question.options[question.correct]}`} /><StudyMarkdown text={question.explanation} /></article>)}</div>}</div>
        : <><div className="quiz-progress"><span>{done} / {questions.length} checked</span><div role="progressbar" aria-label="Quiz progress" aria-valuemin={0} aria-valuemax={questions.length} aria-valuenow={done}><span style={{ width: `${done / questions.length * 100}%` }} /></div></div>
          <article className="quiz-paper"><div className="quiz-question-heading"><span className="eyebrow">QUESTION {index + 1} OF {questions.length}</span><span className="tool-tag">{quiz.settings?.difficulty || "standard"}</span></div><StudyMarkdown className="quiz-question" text={current.question} />
            <div className="answer-choices">{current.options.map((option, i) => <button key={i} type="button" aria-pressed={answers[index] === i} disabled={checked[index]} className={checked[index] && i === current.correct ? "is-correct" : answers[index] === i ? checked[index] ? "is-incorrect" : "is-selected" : ""} onClick={() => update({ answers: { ...answers, [index]: i } })}><span>{String.fromCharCode(65 + i)}</span><StudyMarkdown text={option} />{checked[index] && i === current.correct && <UiIcon name="check" size={18} />}</button>)}</div>
            {hint && !checked[index] && <div className="quiz-hint" role="status"><UiIcon name="bulb" /><StudyMarkdown text={current.hint} /></div>}
            {checked[index] && <div className="answer-explanation" role="status"><strong>{answers[index] === current.correct ? "Correct." : "Here’s how to work it out."}</strong><StudyMarkdown text={current.explanation} /></div>}
            <div className="quiz-check-actions"><button className="text-button" onClick={() => setHint(value => !value)} disabled={checked[index]}><UiIcon name="bulb" />{hint ? "Hide hint" : "Show hint"}</button><button className="quiet-button is-primary" disabled={answers[index] === undefined || checked[index]} onClick={() => update({ checked: { ...checked, [index]: true } })}>Check answer</button></div>
            <footer className="lesson-navigation"><button className="quiet-button" disabled={index === 0} onClick={() => go(index - 1)}><UiIcon name="back" />Previous</button><span>{index + 1} / {questions.length}</span>{index < questions.length - 1 ? <button className="quiet-button" onClick={() => go(index + 1)}>Next<UiIcon name="arrow" /></button> : <button className="quiet-button is-primary" disabled={done !== questions.length} onClick={() => update({ completed: true, finishedAt: Date.now() })}>See results</button>}</footer>
          </article></>}
    {settingsOpen && <StudioDialog title="Quiz settings" onClose={() => setSettingsOpen(false)}><form className="simple-workspace-form" onSubmit={generate}><label>Number of questions<ChoiceMenu value={count} onChange={event => setCount(Number(event.target.value))}>{[5, 10, 15, 20].map(n => <option key={n} value={n}>{n}</option>)}</ChoiceMenu></label><label>Difficulty<ChoiceMenu value={difficulty} onChange={event => setDifficulty(event.target.value)}><option value="foundation">Foundation</option><option value="standard">Standard</option><option value="challenge">Challenge</option></ChoiceMenu></label><label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with study material" /></label><p>These settings apply to a new quiz. Your existing quiz stays in place until generation succeeds.</p>{error && <p role="alert">{error}</p>}<footer><button className="quiet-button" type="button" onClick={() => setSettingsOpen(false)}>Cancel</button><button className="quiet-button is-primary" type="submit" disabled={busy}>{busy ? "Generating…" : "Generate quiz"}</button></footer></form></StudioDialog>}
  </section>;
}
