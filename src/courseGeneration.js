import {parseAiJson} from './aiJson';
import {languageInstruction} from './studyContext';
import {MATH_OUTPUT_GUIDE} from './mathMarkdown';
import {sendStudyRequest} from './studyTools';
import {materialExcerpt} from './assistantRequest';

export function parseLessonPlan(raw) {
  const value = parseAiJson(raw);
  if (!Array.isArray(value) || value.length < 1 || value.length > 16) throw new Error('The AI returned an incomplete course. Your current course is kept.');
  const text = (input, max) => {
    if (typeof input !== 'string' || !input.trim() || input.length > max) throw new Error('The AI returned an incomplete or oversized lesson. Your current course is kept.');
    return input.trim();
  };
  return value.map(unit => {
    if (!unit || !Array.isArray(unit.options) || unit.options.length !== 3 || !Number.isInteger(unit.correct) || unit.correct < 0 || unit.correct > 2) throw new Error('A knowledge check was incomplete. Your current course is kept.');
    return {title:text(unit.title,100),lesson:text(unit.lesson,10000),question:text(unit.question,500),options:unit.options.map(option=>text(option,250)),correct:unit.correct,explanation:text(unit.explanation,1000)};
  });
}

// Partition the complete readable excerpt, rather than sending its beginning
// with every lesson. Prefer paragraph/word boundaries and keep every character.
export function courseSections(source, count) {
  const sections=[]; let start=0;
  for(let i=0;i<count;i++) {
    let end=i===count-1?source.length:Math.round(start+(source.length-start)/(count-i));
    if(end<source.length) {
      const before=source.lastIndexOf('\n\n',end),after=source.indexOf('\n\n',end);
      const boundary=before>start&&(after<0||end-before<=after-end)?before:after;
      const space=source.indexOf(' ',end);
      if(boundary>start&&Math.abs(boundary-end)<180)end=boundary+2;
      else if(space>=0&&space-end<80)end=space+1;
      if(/[\uD800-\uDBFF]/.test(source[end-1]||''))end++;
    }
    sections.push(source.slice(start,end));start=end;
  }
  return sections;
}

export async function generateCourseLessons({source,topic,count,signal,onProgress=()=>{},send=sendStudyRequest}) {
  if(!Number.isInteger(count)||count<1||count>12)throw new Error('Choose between 1 and 12 lessons.');
  const check=()=>{if(signal?.aborted)throw new DOMException('Course generation cancelled.','AbortError');};
  let retries=2; // At most 1 outline + 12 lessons + 2 retries. Normal quotas still apply.
  const request=async(build,parse)=>{
    let compact=false;
    for(;;) {
      check();
      let raw;
      try {raw=await send(build(compact),signal);check();}
      catch(error) {
        if(signal?.aborted||error.code!=='OUTPUT_LIMIT'||!retries||compact)throw error;
        retries--;compact=true;onProgress('Retrying with a shorter response…');continue;
      }
      try{return parse(raw);}
      catch(error) {
        if(!retries||compact)throw error;
        retries--;compact=true;onProgress('Repairing an incomplete response…');
      }
    }
  };
  const language=languageInstruction(source),sections=courseSections(source,count),shortSource=source.length<count*300;
  onProgress('Planning the course…');
  const outline=await request(compact=>({messages:[
    {role:'system',content:`Create a compact course outline for a self-paced course. Return only a JSON array of ${count} objects with title and focus (strings). No lessons or questions yet. ${source?shortSource?'Use the entire short source to plan distinct small lessons. Do not pretend it contains independent chapters.':'Each object must cover its correspondingly numbered source section in order. Do not skip the later sections.':'Progress from foundations to applications of the requested topic.'} ${language} Keep titles under 80 characters and focus under ${compact?100:240} characters.`},
    {role:'user',content:`Topic: ${topic}\n\n${source?shortSource?'SHORT SOURCE:\n'+source:sections.map((part,i)=>`SOURCE SECTION ${i+1}:\n${materialExcerpt(part,'',1200)}`).join('\n\n'):'No source supplied: teach the requested topic, without claiming to quote a file.'}`},
  ],maxTokens:1600}),raw=>{
    const plan=parseAiJson(raw);
    if(!Array.isArray(plan)||plan.length!==count||plan.some(item=>!item||typeof item.title!=='string'||!item.title.trim()||item.title.length>100||typeof item.focus!=='string'||!item.focus.trim()||item.focus.length>500))throw new Error('The course outline was incomplete. Your current course is kept.');
    return plan;
  });
  const units=[];
  for(let index=0;index<count;index++) {
    check();onProgress(`Writing lesson ${index+1} of ${count}…`);
    const lesson=await request(compact=>({messages:[
      {role:'system',content:`Write one complete lesson for this self-paced course. Return only a JSON array containing exactly ONE object: title, lesson (Markdown), question, options (exactly three strings), correct (zero-based integer), explanation. ${language} ${MATH_OUTPUT_GUIDE} Explain the assigned focus with short sections, a worked example and a takeaway. ${compact?'Keep the lesson under 900 characters and the check concise.':'Keep the lesson under 2400 characters and the check concise.'} ${source?'Use the assigned source section as evidence. Label supplemental explanations and do not invent source facts.':'Teach the requested topic without claiming source evidence.'} Do not repeat prior lessons or return the whole course.`},
      {role:'user',content:`Topic: ${topic}\nCourse outline:\n${outline.map((item,i)=>`${i+1}. ${item.title}: ${item.focus}`).join('\n')}\n\nLesson ${index+1} of ${count}: ${outline[index].title}\nFocus: ${outline[index].focus}\n\nAssigned source section ${index+1} (excerpt):\n${materialExcerpt(shortSource?source:sections[index],'',12000)||'(No source supplied; teach only the explicitly requested topic without claiming file evidence.)'}`},
    ],maxTokens:compact?1400:2600}),raw=>{
      const parsed=parseLessonPlan(raw);
      if(parsed.length!==1)throw new Error('The AI returned more than the requested lesson. Your current course is kept.');
      return parsed[0];
    });
    units.push(lesson);
  }
  check();return units;
}
