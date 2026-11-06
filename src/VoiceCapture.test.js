import {fireEvent,render,screen,waitFor} from '@testing-library/react';
import VoiceCapture from './VoiceCapture';
import {sendStudyRequest} from './studyTools';
jest.mock('./podcastAudio',()=>({cachePodcast:jest.fn(async()=>{}),readPodcast:jest.fn(async()=>null)}));
jest.mock('./studyTools',()=>({MAX_AUDIO_BYTES:2800000,downloadStudyFile:jest.fn(),sendStudyRequest:jest.fn()}));
let oldURL;
beforeEach(()=>{oldURL=URL.createObjectURL;URL.createObjectURL=jest.fn(()=> 'blob:test');URL.revokeObjectURL=jest.fn();sendStudyRequest.mockReset();});
afterEach(()=>{URL.createObjectURL=oldURL;});

test('choosing a recording never sends it before the explicit transcription action',async()=>{
 sendStudyRequest.mockResolvedValue('A reviewed lecture transcript.');
 const onSave=jest.fn();render(<VoiceCapture onSave={onSave}/>);
 fireEvent.change(screen.getByLabelText('Choose audio file'),{target:{files:[new File(['local audio bytes'],'lecture.m4a',{type:'audio/mp4'})]}});
 expect(sendStudyRequest).not.toHaveBeenCalled();
 fireEvent.click(screen.getByRole('button',{name:'Transcribe with Groq'}));
 await screen.findByDisplayValue('A reviewed lecture transcript.');
 expect(sendStudyRequest.mock.calls[0][0]).toMatchObject({kind:'audio',filename:'lecture.m4a',provider:'groq',allowFallback:false});
 fireEvent.click(screen.getByRole('button',{name:'Add voice note to workspace'}));
 await waitFor(()=>expect(onSave).toHaveBeenCalledWith(expect.objectContaining({notes:'A reviewed lecture transcript.',file:expect.any(File)})));
});
test('transcription failure preserves the original and leaves download/keep actions available',async()=>{
 sendStudyRequest.mockRejectedValue(new Error('AI is not configured.'));
 const keep=jest.fn();render(<VoiceCapture onSave={()=>{}} onKeepFile={keep}/>);
 const file=new File(['audio'],'lecture.webm',{type:'audio/webm'});
 fireEvent.change(screen.getByLabelText('Choose audio file'),{target:{files:[file]}});
 fireEvent.click(screen.getByRole('button',{name:'Transcribe with Groq'}));
 await screen.findByRole('alert');
 fireEvent.click(screen.getByRole('button',{name:'Keep original as a file'}));
 await waitFor(()=>expect(keep).toHaveBeenCalledWith(file));
 await screen.findByText('Recording stays on this device until you choose Transcribe or save it.');
 expect(screen.getByRole('button',{name:'Download recording'})).toBeEnabled();
});
