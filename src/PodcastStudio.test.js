import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import {useState} from 'react';
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
beforeEach(()=>{store=new Map();global.fetch=jest.fn().mockResolvedValue({ok:true,json:async()=>({aria:{label:'Aria'},echo:{label:'Echo'}})});HTMLMediaElement.prototype.play=jest.fn().mockResolvedValue();HTMLMediaElement.prototype.pause=jest.fn();URL.createObjectURL=jest.fn(()=> 'blob:episode');URL.revokeObjectURL=jest.fn();synthesizeEpisode.mockReset().mockResolvedValue({blob:new Blob(['real audio fixture']),duration:48,turns:episode.turns});});
afterEach(()=>{global.fetch=previous.fetch;HTMLMediaElement.prototype.play=previous.play;HTMLMediaElement.prototype.pause=previous.pause;URL.createObjectURL=previous.create;URL.revokeObjectURL=previous.revoke;});
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
