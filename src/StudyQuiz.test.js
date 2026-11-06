import {fireEvent,render,screen} from '@testing-library/react';
import {useState} from 'react';
import StudyQuiz from './StudyQuiz';
const question={question:'What transfers between objects?',options:['Energy','Nothing','Only colors'],correct:0,lesson:'Objects exchange energy through interactions.',explanation:'Energy can be transferred by work or heating.'};
test('practice questions reveal explanations after answering and retain progress while moving between questions',()=>{
 let value;function Harness(){const [file,setFile]=useState({id:'quiz',studyQuiz:{title:'Energy practice',questions:[question,question]}});value=file;return <StudyQuiz file={file} onUpdate={setFile}/>;}
 render(<Harness/>);expect(screen.queryByText(question.explanation)).not.toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Energy',exact:true}));expect(screen.getByText(question.explanation)).toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Next question'}));expect(value.quizProgress).toMatchObject({active:1,answers:{0:0}});
 fireEvent.click(screen.getByRole('button',{name:'Nothing',exact:true}));expect(screen.getByText('Try another answer.')).toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Previous'}));expect(screen.getByRole('button',{name:'Energy',exact:true})).toHaveAttribute('aria-pressed','true');
});
test('a reopened saved quiz starts at its saved question and answer',()=>{
 render(<StudyQuiz file={{studyQuiz:{title:'Energy practice',questions:[question,question]},quizProgress:{active:1,answers:{1:0}}}} onUpdate={()=>{}}/>);
 expect(screen.getByRole('button',{name:'Question 2'})).toHaveAttribute('aria-current','step');
 expect(screen.getByRole('button',{name:'Energy',exact:true})).toHaveAttribute('aria-pressed','true');
});
