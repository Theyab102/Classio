import {act,fireEvent,render,screen,waitFor} from '@testing-library/react';
import useVoiceCapture from './useVoiceCapture';
import {sendStudyRequest} from './studyTools';
jest.mock('./studyTools',()=>({...jest.requireActual('./studyTools'),sendStudyRequest:jest.fn()}));
function Harness({onTranscript=()=>{}}){const capture=useVoiceCapture(onTranscript);return <><button onClick={capture.start}>Record</button><button onClick={capture.stop}>Stop</button><button onClick={capture.cancel}>Cancel</button><p role="status">{capture.phase}</p>{capture.error&&<p role="alert">{capture.error}</p>}</>;}
const previous={media:navigator.mediaDevices,recorder:window.MediaRecorder,create:URL.createObjectURL,revoke:URL.revokeObjectURL};
let track;
beforeEach(()=>{track={stop:jest.fn()};Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:{getUserMedia:jest.fn().mockResolvedValue({getTracks:()=>[track]})}});URL.createObjectURL=jest.fn(()=> 'blob:recording');URL.revokeObjectURL=jest.fn();sendStudyRequest.mockReset();window.MediaRecorder=class{static isTypeSupported(){return true;}constructor(){this.state='inactive';this.mimeType='audio/webm';}start(){this.state='recording';}stop(){this.state='inactive';this.ondataavailable?.({data:new Blob(['actual encoded audio fixture'],{type:this.mimeType})});this.onstop?.();}};});
afterEach(()=>{Object.defineProperty(navigator,'mediaDevices',{configurable:true,value:previous.media});window.MediaRecorder=previous.recorder;URL.createObjectURL=previous.create;URL.revokeObjectURL=previous.revoke;});
test('a microphone granted after cancellation is stopped and never transcribed',async()=>{
 let grant;navigator.mediaDevices.getUserMedia.mockReturnValue(new Promise(resolve=>grant=resolve));const onTranscript=jest.fn();render(<Harness onTranscript={onTranscript}/>);fireEvent.click(screen.getByText('Record'));expect(screen.getByRole('status')).toHaveTextContent('permission');fireEvent.click(screen.getByText('Cancel'));await act(async()=>grant({getTracks:()=>[track]}));expect(track.stop).toHaveBeenCalled();expect(sendStudyRequest).not.toHaveBeenCalled();expect(onTranscript).not.toHaveBeenCalled();
});
test('permission denial is actionable and recording can be retried without browser speech recognition',async()=>{
 navigator.mediaDevices.getUserMedia.mockRejectedValueOnce(new DOMException('Denied','NotAllowedError'));sendStudyRequest.mockResolvedValue('A reviewed voice message.');const onTranscript=jest.fn();render(<Harness onTranscript={onTranscript}/>);fireEvent.click(screen.getByText('Record'));await screen.findByRole('alert');expect(screen.getByRole('alert')).toHaveTextContent('site settings');fireEvent.click(screen.getByText('Record'));await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('recording'));fireEvent.click(screen.getByText('Stop'));await waitFor(()=>expect(onTranscript).toHaveBeenCalledWith('A reviewed voice message.',expect.any(Blob)));expect(track.stop).toHaveBeenCalled();expect(sendStudyRequest).toHaveBeenCalledTimes(1);expect(sendStudyRequest.mock.calls[0][0]).toMatchObject({kind:'audio',filename:'recording.webm'});
});
test('unmount cancels recording and never sends audio or a late transcript',async()=>{
 const callback=jest.fn(),view=render(<Harness onTranscript={callback}/>);fireEvent.click(screen.getByText('Record'));await waitFor(()=>expect(screen.getByRole('status')).toHaveTextContent('recording'));view.unmount();expect(track.stop).toHaveBeenCalled();expect(sendStudyRequest).not.toHaveBeenCalled();expect(callback).not.toHaveBeenCalled();
});
