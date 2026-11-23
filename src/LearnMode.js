import {startStudyWork,useStudyWork,cancelStudyWork,hasStudyWorkspace,isStudyWorkRunning} from './studyWork';
import {appliedGenerationPatch,useGenerationJob} from './generationJobs';
import GenerationRecovery from './GenerationRecovery';
import {playStudySound} from "./studySounds";
import ChoiceMenu from './ChoiceMenu';
import StudioDialog from './StudioDialog';
import {studySource,responseLanguage} from './studyContext';
import {generateCourseLessons} from './courseGeneration';
import {spokenMathText} from "./mathSpeech";
import { useEffect, useRef, useState } from "react";
import { useStudyActivity } from "./LearningSession";
import StudyMarkdown from "./StudyMarkdown";
import WorkspaceArt from "./WorkspaceArt";
import UiIcon from "./UiIcon";
import "./studyTools.css";

export {parseLessonPlan} from './courseGeneration';
export default function LearnMode({ file, getFileText, onUpdate, onUseAsNotes }) {
  const background=useStudyWork(file.id,"course");
  const [topic, setTopic] = useState(""),[lessonCount,setLessonCount]=useState("auto"),[confirmation,setConfirmation]=useState(null),[progress]=useState("");
  const [active, setActive] = useState(() => Math.max(0, Math.min(file.course?.active || 0, (file.course?.units?.length || 1) - 1)));
  const [choice, setChoice] = useState(() => file.course?.answers?.[file.course?.active || 0] ?? null), [localLoading, setLoading] = useState(false), [error, setError] = useState(""), [speaking, setSpeaking] = useState(false);
  const request = useRef(null), latest = useRef({ file, onUpdate }); latest.current = { file, onUpdate };
  const courseJob=useGenerationJob(file,"course");
  const units = file.course?.units || [], completed = file.course?.completed || [], unit = units[active];
  useStudyActivity(file.id, "Lesson tutor", unit ? `${unit.title}\n${unit.lesson}\nKnowledge check: ${unit.question}` : "The learner is preparing a course.", unit?.title);
  const loading=localLoading||background.running;
  useEffect(() => () => { if(!hasStudyWorkspace()||!isStudyWorkRunning(file.id,"course"))request.current?.abort(); window.speechSynthesis?.cancel(); }, [file.id]);
  const updateCourse = patch => { const current = latest.current; const course = { ...current.file.course, ...patch }; current.onUpdate({ ...current.file, course }); };
  async function generate(event) {
    event?.preventDefault(); if (request.current) return;
    if (units.length) {setConfirmation("replace");return;}
    await generateCourse();
  }
  async function generateCourse(resume) {
    if(resume?.nativeEvent)resume=null;
    if(request.current||background.running)return;
    setConfirmation(null);setLoading(true);setError('');
    const task=startStudyWork({file,kind:'course',fallback:patch=>{const data=latest.current;data.file=patch(data.file);data.onUpdate(data.file);},work:async context=>{
      const source=resume?.input.source??await studySource(file,getFileText,120000);context.check();
      const count=lessonCount==='auto'?Math.min(12,Math.max(3,Math.ceil(source.length/1800))):Number(lessonCount);
      if(!source.trim()&&!topic.trim())throw new Error('No readable study text was found. Paste source text into Notes, or enter a topic for a general course.');
      let checkpoint;const next=await generateCourseLessons({fileId:file.id,resume,onCheckpoint:job=>{checkpoint=job;context.checkpoint(job);},source,topic:topic.trim()||file.name,count,signal:context.signal,onProgress:context.progress});
      context.finishing();context.commit(current=>({...current,...appliedGenerationPatch(current,'course',checkpoint),course:{units:next,completed:[],active:0,answers:{}}}));return next;
    }});request.current=task.controller;
    try{await task.promise;setActive(0);setChoice(null);}
    catch(failure){if(failure.name!=='AbortError')setError(failure.message);}
    finally{if(request.current===task.controller){request.current=null;setLoading(false);}}
  }

  function select(index) {
    window.speechSynthesis?.cancel(); setSpeaking(false); setActive(index); setChoice(file.course?.answers?.[index] ?? null); updateCourse({ active: index });
  }
  function answer(index) {
    playStudySound(index===unit.correct?"correct":"wrong");setChoice(index); updateCourse({ answers: { ...file.course?.answers, [active]: index }, completed: index === unit.correct && !completed.includes(active) ? [...completed, active] : completed });
  }
  function listen() {
    if (speaking) { window.speechSynthesis.cancel(); setSpeaking(false); return; }
    if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) { setError("Read aloud is not supported by this browser."); return; }
    const speech = new SpeechSynthesisUtterance(spokenMathText(`${unit.title}. ${unit.lesson}`));
    speech.lang = responseLanguage(unit.lesson)==="Arabic"?"ar-SA":navigator.language; speech.rate = .95; speech.onend = () => setSpeaking(false); speech.onerror = () => { setSpeaking(false); setError("Read aloud stopped. You can continue reading the lesson."); };
    window.speechSynthesis.cancel(); window.speechSynthesis.speak(speech); setSpeaking(true);
  }
  return <section className="lesson-studio" aria-labelledby="learn-title"><header className="tool-heading"><div><p className="eyebrow">STEP BY STEP</p><h2 id="learn-title">Guided lessons</h2><p>AI explanations with worked examples and helpful questions after each lesson, saved inside Notes.</p></div>
    {unit && <span className="tool-count">{completed.length} / {units.length} complete</span>}</header>
    <form onSubmit={generate} className={`course-generator ${unit ? "is-compact" : ""}`}><label htmlFor="course-topic">Topic<textarea id="course-topic" aria-label="Topic" rows={1} maxLength={2000} placeholder="Optional with source material" value={topic} onInput={event=>{event.currentTarget.style.height="auto";event.currentTarget.style.height=Math.min(240,event.currentTarget.scrollHeight)+"px";}} onChange={event => setTopic(event.target.value)} /></label><label>Lessons<ChoiceMenu aria-label="Number of lessons" value={lessonCount} onChange={event=>setLessonCount(event.target.value)}><option value="auto">Match the material</option>{[3,4,5,6,8,10,12].map(count=><option key={count} value={count}>{count}</option>)}</ChoiceMenu></label><button className="quiet-button is-primary" disabled={loading}><UiIcon name="generate" />{loading ? "Generating lessons…" : unit ? "New course" : "Generate lessons"}</button>{loading&&<button type="button" className="quiet-button" onClick={()=>{cancelStudyWork(file.id,"course");request.current?.abort();request.current=null;setLoading(false);}}>Cancel generation</button>}{unit&&<button type="button" className="quiet-button" onClick={()=>setConfirmation("restart")}>Restart lessons & questions</button>}</form>
    <GenerationRecovery job={courseJob} busy={loading} onResume={generateCourse}/>
    {error && <p role="alert" className="tool-error">{error}</p>}
    {!unit && !loading && <div className="tool-empty"><WorkspaceArt /><h3>Turn a chapter into a clear learning path</h3><p>Use the notes or source in this file, or enter a topic above. Your generated lessons and progress stay with this material.</p></div>}
    {loading && <div className="generation-feedback" role="status"><span className="mini-spinner" />{background.progress||progress || "Reading study material…"} Your current course stays available.</div>}
    {unit && <><div className="course-progress"><div role="progressbar" aria-label="Course progress" aria-valuemin={0} aria-valuemax={units.length} aria-valuenow={completed.length}><span style={{ width: `${completed.length / units.length * 100}%` }} /></div><p role="status">{completed.length} of {units.length} lessons completed</p></div>
      <div className="lesson-layout"><nav className="lesson-outline" aria-label="Course lessons">{units.map((item, index) => <button key={index} onClick={() => select(index)} aria-current={active === index ? "step" : undefined}><span>{completed.includes(index) ? <UiIcon name="check" size={15} /> : String(index + 1).padStart(2, "0")}</span><strong><StudyMarkdown inline text={item.title}/></strong></button>)}</nav>
        <article className="lesson-paper"><div className="lesson-paper-heading"><p className="eyebrow">LESSON {active + 1} OF {units.length}</p>{window.speechSynthesis && <button className="quiet-button" onClick={listen} aria-pressed={speaking}><UiIcon name="volume" />{speaking ? "Stop reading" : "Listen"}</button>}</div><h3><StudyMarkdown inline text={unit.title}/></h3><StudyMarkdown text={unit.lesson} />
          <section className="lesson-check" aria-label="Knowledge check"><p className="eyebrow">CHECK YOUR UNDERSTANDING</p><StudyMarkdown className="lesson-check-question" text={unit.question} /><div className="answer-choices">{unit.options.map((option, index) => <button type="button" key={index} className={choice === index ? index === unit.correct ? "is-correct" : "is-incorrect" : ""} aria-pressed={choice === index} onClick={() => answer(index)}><span>{String.fromCharCode(65 + index)}</span><StudyMarkdown text={option} />{choice === index && <UiIcon name={index === unit.correct ? "check" : "close"} size={17} />}</button>)}</div>
          {choice !== null && <div className="answer-explanation" role="status"><strong>{choice === unit.correct ? "That’s right." : "Take another look."}</strong><StudyMarkdown text={unit.explanation} /></div>}</section>
          <footer className="lesson-navigation"><button className="quiet-button" disabled={active === 0} onClick={() => select(active - 1)}><UiIcon name="back" />Previous</button><span>{active + 1} / {units.length}</span>
            {active < units.length - 1 ? <button className="quiet-button is-primary" onClick={() => select(active + 1)}>Continue<UiIcon name="arrow" /></button> : <button className="quiet-button is-primary" disabled={choice !== unit.correct && !completed.includes(active)} onClick={() => { if (!completed.includes(active)) updateCourse({ completed: [...completed, active] }); }}><UiIcon name="check" />Finish lesson</button>}</footer>
        </article></div>{onUseAsNotes && <button className="text-button lesson-save-notes" onClick={() => onUseAsNotes(units.map(item => `## ${item.title}\n\n${item.lesson}`).join("\n\n"))}><UiIcon name="notes" />Add lessons to notes</button>}</>}
    {confirmation&&<StudioDialog title={confirmation==="replace"?"Generate a new course?":"Restart this course?"} onClose={()=>setConfirmation(null)}><p>{confirmation==="replace"?"Your current course stays available until the new lessons are ready. Your written notes are kept.":"Keep these lessons and start their questions again from the beginning."}</p><div className="dialog-actions"><button className="quiet-button" onClick={()=>setConfirmation(null)}>Cancel</button><button className="quiet-button is-primary" onClick={()=>{if(confirmation==="replace")generateCourse();else{setConfirmation(null);setActive(0);setChoice(null);updateCourse({active:0,answers:{},completed:[]});}}}>{confirmation==="replace"?"Generate new course":"Restart course"}</button></div></StudioDialog>}
  </section>;
}
