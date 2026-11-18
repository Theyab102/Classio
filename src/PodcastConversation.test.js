import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import PodcastConversation from './PodcastConversation';
import {transcribeAudio} from './useVoiceCapture';
import {sendStudyRequest} from './studyTools';
import {synthesizeSpeech} from './podcastAudio';
jest.mock('./useVoiceCapture',()=>({...jest.requireActual('./useVoiceCapture'),transcribeAudio:jest.fn()}));
jest.mock('./studyTools',()=>({...jest.requireActual('./studyTools'),sendStudyRequest:jest.fn()}));
jest.mock('./podcastAudio',()=>({...jest.requireActual('./podcastAudio'),synthesizeSpeech:jest.fn()}));
const episode={id:'one',title:'Forces',turns:[{speaker:'A',section:'Forces',text:'Gravity acts downwards.'}]};
let track,energy;
const previous={media:navigator.mediaDevices,recorder:window.MediaRecorder,context:window.AudioContext,create:URL.createObjectURL,revoke:URL.revokeObjectURL,play:HTMLMediaElement.prototype.play,pause:HTMLMediaElement.prototype.pause};
beforeEach(()=>{
 track={stop:jest.fn()};energy=.08;Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:jest.fn().mockResolvedValue({getTracks:()=>[track]})}});
 window.MediaRecorder=class{static isTypeSupported(){return true;}constructor(){this.state='inactive';this.mimeType='audio/webm';}start(){this.state='recording';}stop(){this.state='inactive';this.ondataavailable?.({data:new Blob(['encoded voice fixture'])});this.onstop?.();}};
 window.AudioContext=class{resume(){return Promise.resolve();}close(){return Promise.resolve();}createMediaStreamSource(){return{connect:()=>{}};}createAnalyser(){return{fftSize:2048,getFloatTimeDomainData:values=>values.fill(energy)};}};
 URL.createObjectURL=jest.fn(()=> 'blob:answer');URL.revokeObjectURL=jest.fn();HTMLMediaElement.prototype.play=jest.fn().mockResolvedValue();HTMLMediaElement.prototype.pause=jest.fn();transcribeAudio.mockReset().mockResolvedValue('Why does gravity act downwards?');sendStudyRequest.mockReset().mockResolvedValue('Gravity attracts objects towards Earth.');synthesizeSpeech.mockReset().mockResolvedValue(new Blob(['speech fixture']));
});
afterEach(()=>{jest.useRealTimers();Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:previous.media});window.MediaRecorder=previous.recorder;window.AudioContext=previous.context;URL.createObjectURL=previous.create;URL.revokeObjectURL=previous.revoke;HTMLMediaElement.prototype.play=previous.play;HTMLMediaElement.prototype.pause=previous.pause;});
test('the same inline button records and sends once; cancel releases the microphone without a dialog',async()=>{
 const onStart=jest.fn(),onFinish=jest.fn();render(<PodcastConversation episode={episode} voice="aria" onStart={onStart} onFinish={onFinish}/>);const button=screen.getByRole('button',{name:'Ask the hosts aloud'});fireEvent.click(button);await screen.findByRole('button',{name:'Send voice question'});expect(screen.getByRole('button',{name:'Send voice question'})).toBe(button);expect(screen.queryByRole('dialog')).not.toBeInTheDocument();expect(onStart).toHaveBeenCalledTimes(1);fireEvent.click(screen.getByRole('button',{name:'Cancel question'}));expect(track.stop).toHaveBeenCalled();expect(transcribeAudio).not.toHaveBeenCalled();expect(onFinish).toHaveBeenCalledWith({forceResume:false});expect(screen.getByRole('button',{name:'Ask the hosts aloud'})).toBe(button);
 fireEvent.click(button);const send=await screen.findByRole('button',{name:'Send voice question'});fireEvent.click(send);fireEvent.click(send);await screen.findByRole('button',{name:'Ask a follow-up aloud'});expect(transcribeAudio).toHaveBeenCalledTimes(1);expect(sendStudyRequest).toHaveBeenCalledTimes(1);expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
});
test('three seconds of actual silence after sustained speech sends automatically',async()=>{
 jest.useFakeTimers();render(<PodcastConversation episode={episode} voice="aria"/>);fireEvent.click(screen.getByRole('button',{name:'Ask the hosts aloud'}));await act(async()=>{});expect(screen.getByRole('button',{name:'Send voice question'})).toBeInTheDocument();act(()=>jest.advanceTimersByTime(400));energy=0;act(()=>jest.advanceTimersByTime(2800));expect(transcribeAudio).not.toHaveBeenCalled();await act(async()=>jest.advanceTimersByTime(400));expect(transcribeAudio).toHaveBeenCalledTimes(1);expect(sendStudyRequest).toHaveBeenCalledTimes(1);expect(track.stop).toHaveBeenCalled();
});
test('late transcription after cancellation cannot send, save or speak an answer',async()=>{
 let complete;transcribeAudio.mockReturnValue(new Promise(resolve=>complete=resolve));const onSave=jest.fn();render(<PodcastConversation episode={episode} voice="aria" onSave={onSave}/>);fireEvent.click(screen.getByRole('button',{name:'Ask the hosts aloud'}));fireEvent.click(await screen.findByRole('button',{name:'Send voice question'}));await waitFor(()=>expect(transcribeAudio).toHaveBeenCalled());fireEvent.click(screen.getByRole('button',{name:'Cancel question'}));await act(async()=>complete('A late transcript.'));expect(sendStudyRequest).not.toHaveBeenCalled();expect(synthesizeSpeech).not.toHaveBeenCalled();expect(onSave).not.toHaveBeenCalled();
});
test('a permission grant arriving after cancel is closed without recording',async()=>{
 let grant;navigator.mediaDevices.getUserMedia.mockReturnValue(new Promise(resolve=>grant=resolve));render(<PodcastConversation episode={episode} voice="aria"/>);fireEvent.click(screen.getByRole('button',{name:'Ask the hosts aloud'}));fireEvent.click(screen.getByRole('button',{name:'Cancel question'}));await act(async()=>grant({getTracks:()=>[track]}));expect(track.stop).toHaveBeenCalled();expect(transcribeAudio).not.toHaveBeenCalled();
});
test('a failed speech service preserves a retry, and completing playback returns to the podcast',async()=>{
 synthesizeSpeech.mockRejectedValueOnce(new Error('Voice service unavailable.'));const onFinish=jest.fn();render(<PodcastConversation episode={episode} voice="aria" onFinish={onFinish}/>);fireEvent.click(screen.getByRole('button',{name:'Ask the hosts aloud'}));fireEvent.click(await screen.findByRole('button',{name:'Send voice question'}));await screen.findByText('Voice service unavailable.');fireEvent.click(screen.getByRole('button',{name:'Retry spoken answer'}));await screen.findByRole('button',{name:'Ask a follow-up aloud'});fireEvent.ended(screen.getByLabelText('Spoken podcast answer'));expect(screen.getByRole('button',{name:'Ask the hosts aloud'})).toBeInTheDocument();expect(onFinish).toHaveBeenCalledWith({forceResume:false});
});
