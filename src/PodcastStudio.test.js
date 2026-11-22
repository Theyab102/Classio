import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {useState} from 'react';
import {act} from '@testing-library/react';
import {sendStudyRequest} from './studyTools';
import {cancelAllPodcastJobs} from './podcastJobs';
jest.mock('./studyTools',()=>({...jest.requireActual('./studyTools'),sendStudyRequest:jest.fn()}));
import PodcastStudio from './PodcastStudio';
import {synthesizeEpisode} from './podcastAudio';
jest.mock('./LearningSession',()=>({useLearningSession:()=>({ask:jest.fn()}),useStudyActivity:()=>{}}));
jest.mock('./pdf',()=>({openPdf:jest.fn(),loadPdfLibrary:jest.fn()}));
jest.mock('./FriendlyAvatar',()=>()=> <span>Host avatar</span>);
jest.mock('./PodcastConversation',()=>({onStart,onFinish})=> <div><button onClick={onStart}>Start inline question</button><button onClick={()=>onFinish({forceResume:false})}>Cancel inline question</button></div>);
jest.mock('./podcastAudio',()=>({...jest.requireActual('./podcastAudio'),synthesizeEpisode:jest.fn()}));
const episode={id:'one',title:'Forces',position:0,turns:[{speaker:'A',section:'Forces',text:'Gravity acts downwards.'}]};
const previous={fetch:global.fetch,play:HTMLMediaElement.prototype.play,pause:HTMLMediaElement.prototype.pause,create:URL.createObjectURL,revoke:URL.revokeObjectURL};
let store;
beforeEach(()=>{cancelAllPodcastJobs();store=new Map();sendStudyRequest.mockReset();global.fetch=jest.fn().mockResolvedValue({ok:true,json:async()=>({aria:{label:'Aria'},echo:{label:'Echo'}})});HTMLMediaElement.prototype.play=jest.fn().mockResolvedValue();HTMLMediaElement.prototype.pause=jest.fn();URL.createObjectURL=jest.fn(()=> 'blob:episode');URL.revokeObjectURL=jest.fn();synthesizeEpisode.mockReset().mockResolvedValue({blob:new Blob(['real audio fixture']),duration:48,turns:episode.turns});});
afterEach(()=>{cancelAllPodcastJobs();global.fetch=previous.fetch;HTMLMediaElement.prototype.play=previous.play;HTMLMediaElement.prototype.pause=previous.pause;URL.createObjectURL=previous.create;URL.revokeObjectURL=previous.revoke;});
function Harness({ready=false,position=0}){if(ready&&!store.size)store.set('saved',new Blob(['saved audio']));const[file,setFile]=useState({id:'notes',name:'Notes',notes:'Gravity',podcastEpisodes:[{...episode,position,...(ready?{audioKey:'saved'}:{})}]});return <PodcastStudio file={file} onUpdate={setFile} audioStore={{get:async key=>store.get(key),save:async(key,blob)=>store.set(key,blob)}}/>;}
test('a transcript-only episode keeps every playback control visible; Play prepares real audio and then plays it',async()=>{
 render(<Harness/>);expect(screen.getByRole('button',{name:'Play podcast'})).toBeEnabled();expect(screen.getByRole('button',{name:'Back 15 seconds'})).toBeDisabled();expect(screen.getByRole('button',{name:'Forward 15 seconds'})).toBeDisabled();expect(screen.getByRole('slider',{name:'Podcast volume'})).toBeEnabled();expect(screen.getByRole('combobox',{name:'Playback speed'})).toBeInTheDocument();fireEvent.click(screen.getByRole('button',{name:'Play podcast'}));await waitFor(()=>expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(1));expect(synthesizeEpisode).toHaveBeenCalledTimes(1);expect(store.size).toBe(1);
});
test('Prepare audio alone does not autoplay; volume and mute operate on the real element',async()=>{
 render(<Harness/>);fireEvent.click(screen.getByRole('button',{name:'Prepare audio'}));await waitFor(()=>expect(document.querySelector('.podcast-player audio')).toBeInTheDocument());const audio=document.querySelector('.podcast-player audio');expect(HTMLMediaElement.prototype.play).not.toHaveBeenCalled();fireEvent.change(screen.getByRole('slider',{name:'Podcast volume'}),{target:{value:'.25'}});expect(audio.volume).toBe(.25);fireEvent.click(screen.getByRole('button',{name:'Mute podcast'}));expect(audio.muted).toBe(true);fireEvent.click(screen.getByRole('button',{name:'Unmute podcast'}));expect(audio.muted).toBe(false);
});
test('skip controls use current playback time and clamp at both ends',async()=>{
 render(<Harness ready/>);await waitFor(()=>expect(document.querySelector('.podcast-player audio')).toBeInTheDocument());const audio=document.querySelector('.podcast-player audio');Object.defineProperty(audio,'duration',{configurable:true,value:48});fireEvent.loadedMetadata(audio);audio.currentTime=4;fireEvent.click(screen.getByRole('button',{name:'Back 15 seconds'}));expect(audio.currentTime).toBe(0);audio.currentTime=44;fireEvent.click(screen.getByRole('button',{name:'Forward 15 seconds'}));expect(audio.currentTime).toBe(48);fireEvent.click(screen.getByRole('button',{name:'Back 15 seconds'}));expect(audio.currentTime).toBe(33);
});
test('asking before metadata loads preserves the saved position when cancelling',async()=>{
 render(<Harness ready position={12}/>);await waitFor(()=>expect(document.querySelector('.podcast-player audio')).toBeInTheDocument());const audio=document.querySelector('.podcast-player audio');fireEvent.click(screen.getByRole('button',{name:'Start inline question'}));Object.defineProperties(audio,{duration:{configurable:true,value:48},readyState:{configurable:true,value:1}});fireEvent.loadedMetadata(audio);fireEvent.click(screen.getByRole('button',{name:'Cancel inline question'}));expect(audio.currentTime).toBe(12);
});
test('script and audio continue across navigation and merge with edits made in another section',async()=>{
 let finishScript,finishAudio;
 sendStudyRequest.mockReturnValue(new Promise(resolve=>finishScript=resolve));
 synthesizeEpisode.mockReturnValue(new Promise(resolve=>finishAudio=resolve));
 function Navigation(){const [file,setFile]=useState({id:'background',name:'Motion notes',notes:'Original notes'}),[visible,setVisible]=useState(true);return <><button onClick={()=>setVisible(value=>!value)}>Switch section</button><button onClick={()=>setFile(current=>({...current,notes:'Edited while podcast runs'}))}>Edit notes</button><button onClick={()=>setFile(current=>({...current,podcastEpisodes:current.podcastEpisodes.map((episode,index)=>index===0?{...episode,voiceQuestions:[{question:"Saved while preparing"}]}:episode)}))}>Save voice question</button><output data-testid="saved-file">{JSON.stringify(file)}</output>{visible&&<PodcastStudio file={file} onUpdate={setFile} audioStore={{get:async key=>store.get(key),save:async(key,blob)=>store.set(key,blob)}}/>}</>;}
 render(<Navigation/>);fireEvent.click(screen.getByRole('button',{name:'Create podcast'}));await waitFor(()=>expect(sendStudyRequest).toHaveBeenCalledTimes(1));
 const signal=sendStudyRequest.mock.calls[0][1];fireEvent.click(screen.getByText('Switch section'));fireEvent.click(screen.getByText('Edit notes'));expect(signal.aborted).toBe(false);
 const turns=Array.from({length:4},(_,i)=>({speaker:i%2?'B':'A',section:'Motion',text:'Gravity changes the vertical velocity.'}));
 await act(async()=>finishScript(JSON.stringify({title:'Motion review',turns})));
 await waitFor(()=>expect(synthesizeEpisode).toHaveBeenCalledTimes(1));
 fireEvent.click(screen.getByText('Save voice question'));
 fireEvent.click(screen.getByText('Switch section'));expect(screen.getByRole('button',{name:'Cancel generation'})).toBeInTheDocument();expect(screen.getByText(/You can switch sections/)).toBeInTheDocument();
 fireEvent.click(screen.getByText('Switch section'));await act(async()=>finishAudio({blob:new Blob(['audio']),duration:32,turns}));
 const saved=JSON.parse(screen.getByTestId('saved-file').textContent);expect(saved.notes).toBe('Edited while podcast runs');expect(saved.podcastEpisodes).toHaveLength(1);expect(saved.podcastEpisodes[0].voiceQuestions).toEqual([{question:"Saved while preparing"}]);expect(store.get(saved.podcastEpisodes[0].audioKey)).toBeInstanceOf(Blob);expect(saved.podcastActiveId).toBe(saved.podcastEpisodes[0].id);
 fireEvent.click(screen.getByText('Switch section'));await waitFor(()=>expect(document.querySelector('.podcast-player audio')).toBeInTheDocument());expect(synthesizeEpisode).toHaveBeenCalledTimes(1);
});
test('creating a podcast preserves the legacy saved transcript in episode history',async()=>{
 const turns=Array.from({length:4},(_,i)=>({speaker:i%2?'B':'A',section:'Forces',text:'Gravity acts downwards.'}));
 sendStudyRequest.mockResolvedValue(JSON.stringify({title:'New episode',turns}));
 function Legacy(){const[file,setFile]=useState({id:'legacy-notes',name:'Legacy notes',notes:'Gravity',podcastScript:'This is the previously saved podcast.'});return <><output data-testid="legacy-file">{JSON.stringify(file)}</output><PodcastStudio file={file} onUpdate={setFile} audioStore={{get:async key=>store.get(key),save:async(key,blob)=>store.set(key,blob)}}/></>;}
 render(<Legacy/>);fireEvent.click(screen.getByRole('button',{name:'Create podcast'}));
 await waitFor(()=>expect(JSON.parse(screen.getByTestId('legacy-file').textContent).podcastEpisodes?.[0].audioKey).toBeTruthy());
 const saved=JSON.parse(screen.getByTestId('legacy-file').textContent);expect(saved.podcastEpisodes).toHaveLength(2);expect(saved.podcastEpisodes[1].id).toBe('legacy');expect(saved.podcastEpisodes[1].turns[0].text).toBe('This is the previously saved podcast.');
});
