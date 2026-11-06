import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import ChatAssistant from './ChatAssistant';
import {requestJson} from './apiClient';
import {setLanguages} from './locale';
jest.mock('./firebase',()=>({auth:{currentUser:{uid:'alice'}},db:{},accountHeaders:jest.fn(async()=>({Authorization:'Bearer test-only-token'}))}));
jest.mock('./apiClient',()=>({requestJson:jest.fn()}));
// PDF workers are exercised in Chromium; this tool-confirmation test never opens a PDF.
jest.mock('./pdf',()=>({loadPdfLibrary:jest.fn(),openPdf:jest.fn()}));
jest.mock('firebase/firestore',()=>({collection:jest.fn(()=>({})),doc:jest.fn(()=>({})),limit:jest.fn(),orderBy:jest.fn(),query:jest.fn(()=>({})),onSnapshot:jest.fn(()=>()=>{}),setDoc:jest.fn(async()=>{})}));
const action={tool:'create_flashcards',arguments:{fileId:'physics',title:'Energy cards',difficulty:'beginner',count:2,instructions:'Use the supplied notes.'}};
let originalFetch;
beforeEach(()=>{originalFetch=global.fetch;global.fetch=jest.fn(async()=>({json:async()=>({providers:{groq:false,gemini:false}})}));sessionStorage.clear();localStorage.clear();requestJson.mockReset();setLanguages({interfaceLanguage:'en',replyLanguage:'auto'},{remote:true});});
afterEach(()=>{global.fetch=originalFetch;});
async function propose(){
 fireEvent.click(screen.getByText('Create from a file'));
 fireEvent.click(screen.getByLabelText('Suggest study actions for my request'));
 fireEvent.change(screen.getByLabelText('Study file'),{target:{value:'physics'}});
 fireEvent.change(screen.getByLabelText('Your message'),{target:{value:'Make flashcards from my energy notes.'}});
 fireEvent.click(screen.getByRole('button',{name:'Send',exact:true}));
 await screen.findByRole('button',{name:'Confirm and create'});
}
test('a real Assistant tool request proposes first, then executes the confirmed validated action',async()=>{
 const open=jest.fn();requestJson.mockResolvedValueOnce({saved:false,message:'',action}).mockResolvedValueOnce({saved:true,file:{id:'cards-result',folderId:'school',name:'Energy cards',defaultTab:'cards'}});
 render(<ChatAssistant onClose={()=>{}} files={[{id:'physics',name:'Physics',notes:'Energy transfers.'}]} onOpenFile={open}/>);
 await propose();expect(requestJson).toHaveBeenCalledTimes(1);expect(requestJson.mock.calls[0][1].body).toMatchObject({tool:'plan_action',arguments:{fileId:'physics',sourceText:'Energy transfers.'}});
 fireEvent.click(screen.getByRole('button',{name:'Confirm and create'}));await screen.findByText('Saved to your account.');
 expect(requestJson.mock.calls[1][1].body).toMatchObject({tool:'create_flashcards',confirmed:true,arguments:{fileId:'physics'}});
 fireEvent.click(screen.getByRole('button',{name:'Open saved file'}));await waitFor(()=>expect(open).toHaveBeenCalledWith(expect.objectContaining({fileId:'cards-result'})));
});
test('finishing a background tool cannot reset a new conversation or overwrite newer messages',async()=>{
 let complete;requestJson.mockResolvedValueOnce({saved:false,message:'',action}).mockImplementationOnce(()=>new Promise(resolve=>{complete=resolve;}));
 render(<ChatAssistant onClose={()=>{}} files={[{id:'physics',name:'Physics',notes:'Energy transfers.'}]}/>);
 await propose();fireEvent.click(screen.getByRole('button',{name:'Confirm and create'}));await waitFor(()=>expect(requestJson).toHaveBeenCalledTimes(2));
 fireEvent.click(screen.getByRole('button',{name:'New chat'}));
 await act(async()=>complete({saved:true,file:{id:'cards-result',folderId:'school',name:'Energy cards',defaultTab:'cards'}}));
 expect(screen.getByRole('heading',{name:'What are you curious about?'})).toBeInTheDocument();
 expect(screen.queryByText('Saved to your account.')).not.toBeInTheDocument();
});
