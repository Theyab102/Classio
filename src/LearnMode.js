import { useEffect, useRef, useState } from 'react';
import { sendStudyRequest, makeStudyId } from './studyTools';
import { auth, accountHeaders } from './firebase';
import { getReplyLanguage, useLocale } from './locale';
import { requestJson } from './apiClient';
import courseSchema from './shared/course-schema.js';
import './studyTools.css';
const {
  parseCourse,
  courseMessages
} = courseSchema;
export const parseLessonPlan = raw => parseCourse(raw);
export default function LearnMode({
  file,
  getFileText,
  onUpdate
}) {
  const { t: tr } = useLocale();
  const [topic, setTopic] = useState(file.course?.title || '');
  const [difficulty, setDifficulty] = useState(file.course?.difficulty || 'beginner');
  const [lessons, setLessons] = useState(4);
  const [objectives, setObjectives] = useState(file.course?.objectives || '');
  const [instructions, setInstructions] = useState('');
  const [active, setActive] = useState(0);
  const [choice, setChoice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const request = useRef(null);
  const requestId = useRef(null);
  const units = file.course?.units || [];
  const completed = file.course?.completed || [];
  const unit = units[active];
  useEffect(() => () => request.current?.abort(), []);
  async function generate(event) {
    event.preventDefault();
    if (request.current) return;
    const controller = new AbortController();
    request.current = controller;
    requestId.current ||= makeStudyId();
    setLoading(true);
    setError('');
    setStatus('Preparing study material…');
    try {
      const source = (file.notes || (await getFileText().catch(() => '')) || '').slice(0, 20000);
      const subject = topic.trim() || file.name;
      if (!source && !topic.trim()) throw new Error('Enter a topic or add study material first.');
      const args = {
        fileId: file.id,
        topic: subject,
        difficulty,
        lessons,
        objectives,
        instructions,
        sourceText: source,
        language: getReplyLanguage()
      };
      let course;
      setStatus('Building lessons and knowledge checks…');
      if (auth.currentUser) {
        let preferences = {};
        try {
          preferences = JSON.parse(localStorage.getItem('classio_ai_preferences') || '{}');
        } catch {}
        const data = await requestJson('/api/agent', {
          headers: await accountHeaders(),
          body: {
            tool: 'create_course',
            arguments: args,
            requestId: requestId.current,
            ...preferences
          },
          signal: controller.signal
        });
        if (data.saved !== true) throw new Error('The course could not be saved. Retry when connected.');
        course = data.course;
        setStatus({key:data.fallback?'Course saved to your account using {provider} (permitted fallback).':'Course saved to your account using {provider}.',values:{provider:data.provider}});
      } else {
        const raw = await sendStudyRequest({
          messages: courseMessages(args, source),
          maxTokens: 7000
        }, controller.signal);
        course = {
          title: subject,
          difficulty,
          objectives,
          units: parseCourse(raw, lessons),
          completed: []
        };
        setStatus('Course created in this guest session. Sign in to save it across devices.');
      }
      if (controller.signal.aborted) return;
      onUpdate({
        ...file,
        course,
        name: file._isBlank ? subject : file.name
      });
      requestId.current = null;
      setActive(0);
      setChoice(null);
    } catch (failure) {
      if (!controller.signal.aborted) {
        setError(failure.message);
        setStatus('');
      }
    } finally {
      request.current = null;
      if (!controller.signal.aborted) setLoading(false);
    }
  }
  function changeRequest(setter, value) {
    setter(value);
    requestId.current = null;
  }
  function answer(index) {
    setChoice(index);
    if (index === unit.correct && !completed.includes(active)) onUpdate({
      ...file,
      course: {
        ...file.course,
        completed: [...completed, active]
      }
    });
  }
  return <section className="study-tool" aria-labelledby="learn-title">
    <p className="entry-kicker">{tr("A clear path forward")}</p><h2 id="learn-title">{tr("Build your course")}</h2>
    <p>{tr("Turn your notes or a topic into focused lessons, examples and practice questions.")}</p>
    <form onSubmit={generate} className="course-form" aria-busy={loading}>
      <label>{tr("Course topic")}<input maxLength={200} placeholder={tr("What would you like to understand?")} value={topic} onChange={event => changeRequest(setTopic, event.target.value)} disabled={loading} /></label>
      <div className="study-tool-actions"><label>{tr("Difficulty")}<select value={difficulty} onChange={event => changeRequest(setDifficulty, event.target.value)} disabled={loading}>{['beginner', 'intermediate', 'advanced'].map(value => <option key={value}>{value}</option>)}</select></label>
      <label>{tr("Lessons")}<select value={lessons} onChange={event => changeRequest(setLessons, Number(event.target.value))} disabled={loading}>{[2, 3, 4, 5, 6].map(value => <option key={value}>{value}</option>)}</select></label></div>
      <label>{tr("Learning objectives")}<textarea maxLength={1500} value={objectives} placeholder={tr("Optional: what should you be able to do?")} onChange={event => changeRequest(setObjectives, event.target.value)} disabled={loading} /></label>
      <details><summary>{tr("Custom instructions")}</summary><label>{tr("Additional guidance")}<textarea maxLength={1500} value={instructions} onChange={event => changeRequest(setInstructions, event.target.value)} disabled={loading} /></label></details>
      <button type="submit" disabled={loading}>{tr(loading ? 'Building course…' : units.length ? 'Build a new course' : error ? 'Retry course generation' : 'Build course')}</button>
    </form>
    {error && <p role="alert">{tr(error)}</p>}{status && <p role="status">{tr(status)}</p>}
    {unit && <>
      <p role="status">{completed.length}{tr(" of ")}{units.length}{tr(" lessons completed")}</p>
      <nav aria-label={tr("Course lessons")} className="study-tool-actions">{units.map((item, index) => <button key={index} type="button" aria-current={index === active ? 'step' : undefined} onClick={() => {
          setActive(index);
          setChoice(null);
          setEditing(false);
        }}>{index + 1}. {item.title}{tr(completed.includes(index) ? ' (done)' : '')}</button>)}</nav>
      <article className="study-lesson"><h3>{unit.title}</h3>
        <button type="button" onClick={() => setEditing(value => !value)}>{tr(editing ? 'Finish editing' : 'Edit lesson')}</button>
        {editing ? <label>{tr("Lesson explanation")}<textarea aria-label={tr("Lesson explanation")} value={unit.lesson} maxLength={5000} onChange={event => onUpdate({
            ...file,
            course: {
              ...file.course,
              units: units.map((item, index) => index === active ? {
                ...item,
                lesson: event.target.value
              } : item)
            }
          })} /></label> : <p className="study-lesson-text">{unit.lesson}</p>}
        <h4>{unit.question}</h4><div className="study-tool-actions">{unit.options.map((option, index) => <button type="button" key={index} aria-pressed={choice === index} onClick={() => answer(index)}>{option}</button>)}</div>
        {choice !== null && <p role="status">{choice === unit.correct ? <>{tr('Correct.')} {unit.explanation}</> : tr('Try another answer.')}</p>}
        {completed.includes(active) && active < units.length - 1 && <button type="button" onClick={() => {
          setActive(active + 1);
          setChoice(null);
        }}>{tr("Next lesson")}</button>}
      </article>
    </>}
  </section>;
}
