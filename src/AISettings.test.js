import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import AISettings from './AISettings';
import {aiMode,setAiMode} from './aiPreferences';
import {loadLocalModel,disableLocalAi} from './localAi';
jest.mock('./localAi',()=>({useLocalAi:()=>({llm:false,embeddings:false,loading:false,progress:0,error:''}),localCompatibility:()=>({webgpu:false,memoryGB:2,canLoad:false}),loadLocalModel:jest.fn(),disableLocalAi:jest.fn(),clearLocalModels:jest.fn()}));
beforeEach(()=>{localStorage.clear();jest.clearAllMocks();global.fetch=jest.fn(async()=>({json:async()=>({aiConfigured:true,providers:[{id:'groq',status:'available'}],browserLocalEnabled:true,freeOnly:true})}));});
test('settings show only permitted providers without credential fields and preserve a hosted choice when local AI is disabled',async()=>{
 render(<AISettings/>);await screen.findByText(/Free-only operation/);fireEvent.click(screen.getByRole('combobox',{name:'AI mode'}));expect(screen.queryByRole('option',{name:/Gemini|Cloudflare|Private server/})).not.toBeInTheDocument();fireEvent.click(screen.getByRole('option',{name:'Groq',exact:true}));expect(aiMode()).toBe('groq');fireEvent.click(screen.getByText('On-device AI and semantic search'));fireEvent.click(screen.getByRole('button',{name:'Disable local AI'}));expect(aiMode()).toBe('groq');expect(disableLocalAi).toHaveBeenCalledTimes(1);expect(document.body.textContent).not.toMatch(/packup|API_KEY|CLASSIO_SESSION_SECRET/);
});
test('model downloads require opt-in; unsupported graphics cannot load the local LLM and local mode can be disabled',async()=>{
 setAiMode('browser-local');render(<AISettings/>);await screen.findByText(/Free-only operation/);fireEvent.click(screen.getByText('On-device AI and semantic search'));expect(screen.getByRole('button',{name:'Enable semantic search'})).toBeDisabled();expect(loadLocalModel).not.toHaveBeenCalled();fireEvent.click(screen.getByRole('checkbox',{name:'Allow model downloads on this device'}));expect(screen.getByRole('button',{name:'Download local AI'})).toBeDisabled();fireEvent.click(screen.getByRole('button',{name:'Enable semantic search'}));await waitFor(()=>expect(loadLocalModel).toHaveBeenCalledWith('embeddings','english'));fireEvent.click(screen.getByRole('button',{name:'Disable local AI'}));expect(aiMode()).toBe('auto');
});
