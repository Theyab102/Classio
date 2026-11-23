import {lessonReference,lessonCoverage,omitLogistics,allowsAdvancedMethods,validateLessonCoverage} from './lessonCoverage';
import {makeSourceBundle} from './sourceBundle';
const pack=Array.from({length:33},(_,i)=>`[Page ${i+1}]\nLesson ${i+1}: Polynomial graphs\nQuestion ${i+1}: Describe the graph shape for this lesson.`).join('\n\n');
test('all 33 pages remain in the teaching plan and an answer key is a reference, not a second course',()=>{
 const source=makeSourceBundle([{name:'Grade 11 revision pack.pdf',text:pack},{name:'Answer key.pdf',text:pack.replaceAll('Question','Answer')}]);
 const plan=lessonCoverage(source);expect(plan).toHaveLength(33);expect(plan.map(item=>item.page)).toEqual(Array.from({length:33},(_,i)=>i+1));expect(plan.at(-1).source).toContain('Question 33');
});
test('logistics are omitted without discarding a mixed line with actual lesson material',()=>{
 expect(omitLogistics('Exam date: Monday\nTime allowed: 90 minutes\nLesson 2: Quadratics\nQuestion 1: Solve x^2=4')).toBe('Lesson 2: Quadratics\nQuestion 1: Solve x^2=4');
 expect(omitLogistics('Exam date: Monday; Lesson 2 includes f(x)=x^2')).toContain('f(x)=x^2');
});
test('grade alone does not authorize calculus and a no-limits instruction takes priority',()=>{
 expect(allowsAdvancedMethods('Grade 11. Lesson 1: End behavior.')).toBe(false);expect(allowsAdvancedMethods('Lesson 1: Limits and continuity')).toBe(true);expect(allowsAdvancedMethods('Lesson 1: Limits and continuity','Explain without limits or calculus')).toBe(false);
});
test('each source question must be covered; reusable teaching definitions do not falsely stop later pages',()=>{
 const first='Question 1\nA correct, clear explanation with a common definition that can also be needed in a later question.';
 expect(()=>validateLessonCoverage('Question 1\nAn explanation of the first problem, with steps and a worked example.',[],'Question 1: First.\nQuestion 2: Second.')).toThrow('skipped');
 expect(validateLessonCoverage('Question 2\nA correct, clear explanation with a common definition that can also be needed in a later question.',[first],'Question 2: Solve.')).toContain('Question 2');expect(()=>validateLessonCoverage(first,[first],'Question 1')).toThrow('repeated');
});

test('a dense page is divided at question boundaries, with lesson context kept in every part',()=>{
 const plan=lessonCoverage('[Page 1]\nLesson: Algebra\n'+Array.from({length:8},(_,i)=>`Question ${i+1}: Solve the equation.`).join('\n'));
 expect(plan).toHaveLength(3);expect(plan.every(unit=>unit.page===1&&unit.source.includes('Lesson: Algebra'))).toBe(true);expect(plan[0].source).not.toContain('Question 4');expect(plan[2].source).toContain('Question 8');
});
test('a later revision question retrieves its matching answer instead of only the start of a long key',()=>{
 const source=makeSourceBundle([{name:'Pack.pdf',text:'Question 33: Find the roots of the quadratic.'},{name:'Answer key.txt',text:Array.from({length:80},(_,i)=>`Question ${i+1}: ${i===32?'Unique worked solution: roots 1 and 2.':'Other answer evidence. '.repeat(15)}`).join('\n')}]);
 const reference=lessonReference(source,{file:'Pack.pdf',source:'Question 33: Find the roots of the quadratic.'});expect(reference).toContain('Unique worked solution: roots 1 and 2.');expect(reference).toContain('Answer key.txt');expect(reference.length).toBeLessThanOrEqual(6000);
});
