import {useState} from 'react';
import {useLocale} from './locale';
export default function StudyQuiz({file,onUpdate}){
  const {t}=useLocale();
  const quiz=file.studyQuiz,questions=quiz.questions,answers=file.quizProgress?.answers||{};
  const [active,setActive]=useState(Math.min(file.quizProgress?.active||0,questions.length-1));
  const question=questions[active],choice=answers[active];
  const score=questions.filter((item,index)=>answers[index]===item.correct).length;
  function move(index){setActive(index);onUpdate({...file,quizProgress:{answers,active:index}});}
  return <section className="study-tool" aria-labelledby="practice-quiz-title"><p className="entry-kicker">{t('Practice quiz')}</p><h2 id="practice-quiz-title">{quiz.title}</h2><p>{t('Review a question, choose an answer, then explore the explanation.')}</p><nav className="study-tool-actions" aria-label={t('Quiz questions')}>{questions.map((item,index)=><button key={index} type="button" aria-label={t('Question')+' '+(index+1)} aria-current={index===active?'step':undefined} onClick={()=>move(index)}>{index+1}{answers[index]===item.correct?' ✓':''}</button>)}</nav><article className="study-lesson"><h3>{question.question}</h3><div className="study-tool-actions">{question.options.map((option,index)=><button key={index} type="button" aria-pressed={choice===index} onClick={()=>onUpdate({...file,quizProgress:{active,answers:{...answers,[active]:index}}})}>{option}</button>)}</div>{choice!==undefined&&<div role="status"><strong>{t(choice===question.correct?'Correct.':'Try another answer.')}</strong><p>{question.explanation}</p><details><summary>{t('Review the concept')}</summary><p>{question.lesson}</p></details></div>}</article><p role="status">{t('Correct answers')}: {score} / {questions.length}</p><div className="study-tool-actions"><button type="button" disabled={active===0} onClick={()=>move(active-1)}>{t('Previous')}</button><button type="button" disabled={active===questions.length-1} onClick={()=>move(active+1)}>{t('Next question')}</button></div></section>;
}
