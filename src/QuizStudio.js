import {generateStudySet} from './checkpointGenerators';
import {checkpointPatch,appliedGenerationPatch,useGenerationJob} from './generationJobs';
import GenerationRecovery from './GenerationRecovery';
import {playStudySound} from "./studySounds";
import {studySource} from './studyContext';
import {parseAiJson} from './aiJson';
import ChoiceMenu from "./ChoiceMenu";
import { useEffect, useRef, useState } from "react";
import { useStudyActivity } from "./LearningSession";
import StudioDialog from "./StudioDialog";
import StudyMarkdown from "./StudyMarkdown";
import WorkspaceArt from "./WorkspaceArt";
import UiIcon from "./UiIcon";

export function parseQuiz(raw) {
  const value = parseAiJson(raw);
  const rows = Array.isArray(value) ? value : value.questions;
  if (!Array.isArray(rows) || !rows.length || rows.length > 50) throw new Error("The AI returned an incomplete quiz. Try again.");
  return rows.map(row => {
    if(row?.type==='short-answer'){
      if(['question','answer','hint','explanation'].some(key=>typeof row[key]!=='string'||!row[key].trim()||row[key].length>1600)||!Array.isArray(row.acceptedAnswers)||row.acceptedAnswers.length<1||row.acceptedAnswers.length>5||row.acceptedAnswers.some(answer=>typeof answer!=='string'||!answer.trim()||answer.length>500))throw new Error('A short-answer question is incomplete. Saved questions are kept.');
      return{type:'short-answer',question:row.question.trim(),answer:row.answer.trim(),acceptedAnswers:[...new Set([row.answer.trim(),...row.acceptedAnswers.map(answer=>answer.trim())])],hint:row.hint.trim(),explanation:row.explanation.trim()};
    }
    if (!row || typeof row.question !== "string" || !row.question.trim() || !Array.isArray(row.options) || row.options.length < 3 || row.options.length > 4 || row.options.some(text => typeof text !== "string" || !text.trim()) || !Number.isInteger(row.correct) || row.correct < 0 || row.correct >= row.options.length || typeof row.explanation !== "string" || !row.explanation.trim() || typeof row.hint !== "string" || !row.hint.trim()) throw new Error("A quiz question was incomplete. Try generating again.");
    return { question: row.question.trim().slice(0, 1500), options: row.options.map(text => text.trim().slice(0, 500)), correct: row.correct, hint: row.hint.slice(0, 800), explanation: row.explanation.slice(0, 1600) };
  });
}
export function isQuizCorrect(question,answer){const normalize=value=>String(value??'').normalize('NFKC').toLocaleLowerCase().replace(/\s+/g,' ').trim().replace(/[.!؟。]$/,'');return question.type==='short-answer'?typeof answer==='string'&&[question.answer,...(question.acceptedAnswers||[])].some(expected=>normalize(expected)===normalize(answer)):answer===question.correct;}
export default function QuizStudio({ file, onUpdate, getFileText }) {
  const [settingsOpen, setSettingsOpen] = useState(false), [count, setCount] = useState(String(file.quiz?.settings?.count ?? 0)), [difficulty, setDifficulty] = useState(file.quiz?.settings?.difficulty || "standard"), [topic, setTopic] = useState("");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [hint, setHint] = useState(false), [review, setReview] = useState(false);
  const [questionType,setQuestionType]=useState(file.quiz?.settings?.type||"multiple-choice");
  const quizJob=useGenerationJob(file,"quiz");
  const [confirmation,setConfirmation]=useState(false),[progress,setProgress]=useState("");
  const pending = useRef(null), latest = useRef({ file, onUpdate }); latest.current = { file, onUpdate };
  const quiz = file.quiz, questions = quiz?.questions || [], index = Math.min(quiz?.active || 0, Math.max(questions.length - 1, 0)), current = questions[index], answers = quiz?.answers || {}, checked = quiz?.checked || {};
  const correct = questions.filter((question, i) => checked[i] && isQuizCorrect(question,answers[i])).length;
  const done = Object.keys(checked).filter(key => checked[key]).length;
  useStudyActivity(file.id, "Quiz hints", current ? `${current.question}\nChoices:\n${(current.options||[]).join("\n")}${checked[index] ? `\nExplanation: ${current.explanation}` : "\nGive a hint first without revealing the answer."}` : "The learner is preparing a quiz.");
  useEffect(() => () => pending.current?.abort(), []);
  const update = patch => { const data = latest.current; data.onUpdate({ ...data.file, quiz: { ...data.file.quiz, ...patch } }); };
  function closeSettings(){pending.current?.abort();pending.current=null;setBusy(false);setSettingsOpen(false);}
  async function generate(event) {
    event?.preventDefault(); if (busy) return;
    if(!Number.isInteger(Number(count))||Number(count)<1||Number(count)>50){setError("Choose a whole number from 1 to 50 questions.");return;}
    if (questions.length) {setConfirmation(true);return;}
    await generateQuiz();
  }
  async function generateQuiz(resume) {
    if(resume?.nativeEvent)resume=null;
    if(pending.current)return;
    setConfirmation(false);
    const controller=new AbortController();pending.current=controller;setBusy(true);setError("");
    try {
      const source=resume?.input.source??await studySource(file,getFileText);
      if(!source&&!topic.trim()&&!resume)throw new Error("Add study material or enter a topic before generating a quiz.");
      let checkpoint;
      const input=resume?.input||{source,topic:topic.trim()||file.name,count:Number(count),difficulty,type:questionType};
      const next=await generateStudySet({fileId:file.id,kind:'quiz',input,resume,parse:parseQuiz,signal:controller.signal,onProgress:setProgress,onCheckpoint:job=>{checkpoint=job;const data=latest.current;data.file=checkpointPatch(data.file,'quiz',job);data.onUpdate(data.file);}});
      if(controller.signal.aborted)return;
      const data=latest.current;data.onUpdate({...data.file,...appliedGenerationPatch(data.file,'quiz',checkpoint),quiz:{id:checkpoint.id,questions:next,answers:{},checked:{},active:0,completed:false,settings:{count:input.count,difficulty:input.difficulty,type:input.type}}});
      setSettingsOpen(false);setReview(false);setHint(false);
    }catch(failure){if(!controller.signal.aborted)setError(failure.message);}
    finally{if(pending.current===controller){setBusy(false);pending.current=null;}}
  }
  const go = next => { update({ active: next }); setHint(false); };
  const finished = quiz?.completed;
  return <section className="quiz-studio"><header className="tool-heading"><div><p className="eyebrow">ACTIVE RECALL</p><h2>Quizzes</h2><p>Check what you understand, then work through the explanations.</p></div><button className="quiet-button" onClick={() => setSettingsOpen(true)}><UiIcon name="settings" />Quiz settings</button></header>
    {current&&<div className="quiz-regenerate studio-controls"><label>Questions<input type="number" aria-label="Number of questions" min="0" max="50" step="1" value={count} onChange={event=>setCount(event.target.value)}/></label><button className="quiet-button is-primary" disabled={busy} onClick={generate}><UiIcon name="generate"/>{busy?"Generating quiz…":"Generate new questions"}</button></div>}
    <GenerationRecovery job={quizJob} busy={busy} onResume={generateQuiz}/>
    {busy&&<div className="generation-feedback" role="status"><span className="mini-spinner"/>{progress}<button className="quiet-button" onClick={()=>{pending.current?.abort();pending.current=null;setBusy(false);}}>Cancel generation</button></div>}
    {error && <p className="tool-error" role="alert">{error}</p>}
    {!current ? <><form onSubmit={generate} className="course-generator"><label>Questions<input type="number" aria-label="Number of questions" min="0" max="50" step="1" value={count} onChange={event=>setCount(event.target.value)}/></label><label>Question type<ChoiceMenu aria-label="Question type" value={questionType} onChange={event=>setQuestionType(event.target.value)}><option value="multiple-choice">Multiple choice</option><option value="short-answer">Short answer</option></ChoiceMenu></label><label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with study material" /></label><button className="quiet-button is-primary" disabled={busy}><UiIcon name="generate" />{busy ? "Generating quiz…" : "Generate quiz"}</button></form>
      {!busy && <div className="tool-empty"><WorkspaceArt /><h3>Put your understanding into practice</h3><p>Generate questions from this material. Each question includes a hint and an explanation, and your answers stay with this file.</p></div>}</>
      : finished ? <div className="quiz-results"><span className="result-icon"><UiIcon name="check" size={32} /></span><p className="eyebrow">QUIZ COMPLETE</p><h3>{correct} of {questions.length} correct</h3><p>{done < questions.length ? `${questions.length - done} questions were not checked.` : "Use the explanations to revisit anything that felt uncertain."}</p><div className="studio-controls"><button className="quiet-button is-primary" onClick={() => setReview(value => !value)}>{review ? "Hide explanations" : "Review answers"}</button><button className="quiet-button" onClick={() => { update({ answers: {}, checked: {}, active: 0, completed: false }); setReview(false); }}>Try again</button></div>
        {review && <div className="quiz-answer-review">{questions.map((question, i) => <article key={i}><StudyMarkdown text={`${i + 1}. ${question.question}`} /><StudyMarkdown text={`Your answer: ${question.type==="short-answer"?answers[i]||"Not answered":question.options[answers[i]]||"Not answered"}`} /><StudyMarkdown text={`Correct answer: ${question.type==="short-answer"?question.answer:question.options[question.correct]}`} /><StudyMarkdown text={question.explanation} /></article>)}</div>}</div>
        : <><div className="quiz-progress"><span>{done} / {questions.length} checked</span><div role="progressbar" aria-label="Quiz progress" aria-valuemin={0} aria-valuemax={questions.length} aria-valuenow={done}><span style={{ width: `${done / questions.length * 100}%` }} /></div></div>
          <article className="quiz-paper"><div className="quiz-question-heading"><span className="eyebrow">QUESTION {index + 1} OF {questions.length}</span><span className="tool-tag">{quiz.settings?.difficulty || "standard"}</span></div><StudyMarkdown className="quiz-question" text={current.question} />
            {current.type==="short-answer"?<label className="quiz-short-answer">Your answer<input aria-label="Your answer" maxLength={500} value={answers[index]||""} disabled={checked[index]} onChange={event=>update({answers:{...answers,[index]:event.target.value}})}/><small>Use a short phrase. After checking, compare with the expected answer.</small></label>:<div className="answer-choices">{current.options.map((option, i) => <button key={i} type="button" aria-pressed={answers[index] === i} disabled={checked[index]} className={checked[index] && i === current.correct ? "is-correct" : answers[index] === i ? checked[index] ? "is-incorrect" : "is-selected" : ""} onClick={() => update({ answers: { ...answers, [index]: i } })}><span>{String.fromCharCode(65 + i)}</span><StudyMarkdown text={option} />{checked[index] && i === current.correct && <UiIcon name="check" size={18} />}</button>)}</div>}
            {hint && !checked[index] && <div className="quiz-hint" role="status"><UiIcon name="bulb" /><StudyMarkdown text={current.hint} /></div>}
            {checked[index] && <div className="answer-explanation" role="status"><strong>{isQuizCorrect(current,answers[index]) ? "Correct." : "Here’s how to work it out."}</strong>{current.type==="short-answer"&&<StudyMarkdown text={`Expected answer: ${current.answer}`}/>}<StudyMarkdown text={current.explanation} /></div>}
            <div className="quiz-check-actions"><button className="text-button" onClick={() => setHint(value => !value)} disabled={checked[index]}><UiIcon name="bulb" />{hint ? "Hide hint" : "Show hint"}</button><button className="quiet-button is-primary" disabled={answers[index] === undefined || (current.type==="short-answer"&&!String(answers[index]).trim()) || checked[index]} onClick={() => {playStudySound(isQuizCorrect(current,answers[index])?"correct":"wrong");update({ checked: { ...checked, [index]: true } });}}>Check answer</button></div>
            <footer className="lesson-navigation"><button className="quiet-button" disabled={index === 0} onClick={() => go(index - 1)}><UiIcon name="back" />Previous</button><span>{index + 1} / {questions.length}</span>{index < questions.length - 1 ? <button className="quiet-button" onClick={() => go(index + 1)}>Next<UiIcon name="arrow" /></button> : <button className="quiet-button is-primary" disabled={done !== questions.length} onClick={() => update({ completed: true, finishedAt: Date.now() })}>See results</button>}</footer>
          </article></>}
    {settingsOpen && <StudioDialog title="Quiz settings" onClose={closeSettings}><form className="simple-workspace-form" onSubmit={generate}><label>Number of questions<input type="number" aria-label="Number of questions" min="0" max="50" step="1" value={count} onChange={event=>setCount(event.target.value)}/></label><label>Difficulty<ChoiceMenu value={difficulty} onChange={event => setDifficulty(event.target.value)}><option value="foundation">Foundation</option><option value="standard">Standard</option><option value="challenge">Challenge</option></ChoiceMenu></label><label>Question type<ChoiceMenu aria-label="Question type" value={questionType} onChange={event=>setQuestionType(event.target.value)}><option value="multiple-choice">Multiple choice</option><option value="short-answer">Short answer</option></ChoiceMenu></label><label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with study material" /></label><p>These settings apply to a new quiz. Your existing quiz stays in place until generation succeeds.</p>{error && <p role="alert">{error}</p>}<footer><button className="quiet-button" type="button" onClick={closeSettings}>Cancel</button><button className="quiet-button is-primary" type="submit" disabled={busy}>{busy ? "Generating…" : "Generate quiz"}</button></footer></form></StudioDialog>}
    {confirmation&&<StudioDialog title="Generate new questions?" onClose={()=>setConfirmation(false)}><p>Your current attempt is replaced only when the new quiz is ready. Your other study work is kept.</p><div className="dialog-actions"><button className="quiet-button" onClick={()=>setConfirmation(false)}>Cancel</button><button className="quiet-button is-primary" onClick={generateQuiz}>Generate new quiz</button></div></StudioDialog>}
  </section>;
}
