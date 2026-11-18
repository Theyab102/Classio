import 'fake-indexeddb/auto';
import {setAiIdentityProvider} from './aiAccess';
beforeEach(()=>setAiIdentityProvider(()=>({uid:'fixture-'+expect.getState().currentTestName})));
import FlashcardStudio from './FlashcardStudio';
import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import {useState} from 'react';
import LearnMode from './LearnMode';
import QuizStudio from './QuizStudio';
import {sendStudyRequest} from './studyTools';
jest.mock('./studyTools',()=>({...jest.requireActual('./studyTools'),sendStudyRequest:jest.fn()}));
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
const q={question:'Question?',options:['One','Two','Three','Four'],correct:0,hint:'Think it through',explanation:'One is correct'};
const unit={title:'Saved lesson',lesson:'A source-grounded explanation',question:'Check?',options:['One','Two','Three'],correct:0,explanation:'Because one'};
function Harness({Component,initial,changed,source}){const[file,setFile]=useState(initial);return <Component file={file} getFileText={async()=>source||'Original study material.'} onUpdate={next=>{changed(next);setFile(next);}}/>;}
beforeEach(()=>{jest.clearAllMocks();localStorage.clear();});
test('quizzes accept a cleared typed count and generate an arbitrary amount in bounded batches',async()=>{
 const changed=jest.fn();sendStudyRequest.mockResolvedValueOnce(JSON.stringify(Array.from({length:8},(_,i)=>({...q,question:"Question "+i})))).mockResolvedValueOnce(JSON.stringify(Array.from({length:5},(_,i)=>({...q,question:"Question "+(i+8)}))));
 render(<Harness Component={QuizStudio} initial={{id:'quiz',name:'Chapter'}} changed={changed}/>);
 const input=screen.getByRole('spinbutton',{name:'Number of questions'});expect(input).toHaveValue(0);fireEvent.change(input,{target:{value:''}});expect(input).toHaveValue(null);fireEvent.change(input,{target:{value:'13'}});
 fireEvent.click(screen.getByRole('button',{name:'Generate quiz',exact:true}));await waitFor(()=>expect(changed.mock.calls.at(-1)?.[0].quiz?.questions).toHaveLength(13));
 expect(changed.mock.calls.at(-1)[0].quiz.questions).toHaveLength(13);expect(sendStudyRequest).toHaveBeenCalledTimes(2);
});
test('regeneration uses an app dialog and keeps the old attempt if a later batch fails',async()=>{
 const changed=jest.fn(),confirm=jest.spyOn(window,'confirm');sendStudyRequest.mockResolvedValueOnce(JSON.stringify(Array.from({length:8},(_,i)=>({...q,question:"Question "+i})))).mockRejectedValue(new Error('Provider temporarily unavailable'));
 render(<Harness Component={QuizStudio} initial={{id:'q',name:'Saved',quiz:{questions:[q],answers:{0:1},checked:{},settings:{count:13}}}} changed={changed}/>);
 fireEvent.click(screen.getByRole('button',{name:'Generate new questions'}));expect(screen.getByRole('dialog',{name:'Generate new questions?'})).toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:'Generate new quiz',exact:true}));
 await screen.findByText('Provider temporarily unavailable');expect(changed.mock.calls.at(-1)[0].quiz.questions).toEqual([q]);expect(changed.mock.calls.at(-1)[0].generationJobs.quiz.sections).toHaveLength(1);expect(confirm).not.toHaveBeenCalled();confirm.mockRestore();
});
test('restarting guided lessons resets checks and keeps lesson text and written notes',()=>{
 const changed=jest.fn();render(<Harness Component={LearnMode} initial={{id:'lesson',name:'Chapter',notes:'Keep notes',course:{units:[unit,unit],answers:{0:0},completed:[0],active:1}}} changed={changed}/>);
 fireEvent.click(screen.getByRole('button',{name:'Restart lessons & questions'}));fireEvent.click(screen.getByRole('button',{name:'Restart course',exact:true}));
 expect(changed.mock.calls.at(-1)[0]).toMatchObject({notes:'Keep notes',course:{active:0,answers:{},completed:[],units:[unit,unit]}});
});
test('Arabic lessons send the original text and an explicit Arabic instruction',async()=>{
 const changed=jest.fn();sendStudyRequest.mockResolvedValueOnce(JSON.stringify([1,2,3].map(i=>({title:'الدرس '+i,focus:'الخلية النباتية'})))).mockResolvedValue(JSON.stringify([unit]));render(<Harness Component={LearnMode} initial={{id:'ar',name:'Old English title',notes:'An unrelated old summary'}} source='يتناول هذا الفصل الخلية النباتية ومكوناتها ووظيفة كل جزء منها.' changed={changed}/>);
 fireEvent.click(screen.getByRole('button',{name:'Generate lessons',exact:true}));await waitFor(()=>expect(sendStudyRequest).toHaveBeenCalled());expect(sendStudyRequest.mock.calls[0][0].messages[0].content).toContain('in Arabic');expect(sendStudyRequest.mock.calls[0][0].messages[1].content).toContain('الخلية النباتية');
});
test('a filename without readable material never becomes an invented course',async()=>{
 const changed=jest.fn();render(<LearnMode file={{id:'scan',name:'Arabic biology.pdf'}} getFileText={async()=>''} onUpdate={changed}/>);
 fireEvent.click(screen.getByRole('button',{name:'Generate lessons',exact:true}));await screen.findByText(/No readable study text was found/);
 expect(sendStudyRequest).not.toHaveBeenCalled();expect(changed).not.toHaveBeenCalled();
});

test('cancelling flashcard generation prevents a late response from changing the saved set',async()=>{let resolve;const changed=jest.fn();sendStudyRequest.mockImplementationOnce(()=>new Promise(done=>{resolve=done;}));render(<Harness Component={FlashcardStudio} initial={{id:'cancel',name:'Material',studyCards:[{id:'old',question:'Keep?',answer:'Keep.'}]}} changed={changed}/>);fireEvent.click(screen.getByRole('button',{name:'Generate cards',exact:true}));fireEvent.change(screen.getByRole('spinbutton',{name:'Number of cards'}),{target:{value:'1'}});fireEvent.submit(screen.getByRole('dialog').querySelector('form'));await waitFor(()=>expect(sendStudyRequest).toHaveBeenCalled());fireEvent.click(screen.getByRole('button',{name:'Cancel',exact:true}));await act(async()=>{resolve(JSON.stringify([{question:'Late',answer:'No replacement'}]));});expect(changed.mock.calls.every(([next])=>next.studyCards.length===1&&next.studyCards[0].id==="old")).toBe(true);expect(document.querySelector('.flashcard-face.is-front')).toHaveTextContent('Keep?');});
