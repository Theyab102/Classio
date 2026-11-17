import {useEffect,useRef,useState} from 'react';
import useVoiceCapture from './useVoiceCapture';
import {sendStudyRequest} from './studyTools';
import {languageInstruction} from './studyContext';
import UiIcon from './UiIcon';
export default function VoiceRecorder({onSave,onSaveAudio}){
  const[text,setText]=useState(''),[notes,setNotes]=useState(''),[blob,setBlob]=useState(null),[busy,setBusy]=useState(false),[error,setError]=useState('');
  const pending=useRef(null),upload=useRef(null);
  const capture=useVoiceCapture((transcript,audio)=>{setText(transcript);setNotes('');setBlob(audio);},{maxSeconds:600});
  useEffect(()=>()=>pending.current?.abort(),[]);
  async function organize(){const controller=new AbortController();pending.current=controller;setBusy(true);setError('');try{const result=await sendStudyRequest({task:'notes',messages:[{role:'system',content:'Organize this transcript into clear study notes. Keep facts, equations and uncertainty; do not invent details. '+languageInstruction(text)},{role:'user',content:text}],maxTokens:3500},controller.signal);if(!controller.signal.aborted)setNotes(result);}catch(failure){if(!controller.signal.aborted)setError(failure.message);}finally{if(!controller.signal.aborted)setBusy(false);}}
  return <div className="voice-recorder"><div className="voice-recorder-start"><UiIcon name="mic" size={32}/><h3>{capture.recording?'Recording…':'Tap to start recording'}</h3><p>Record a lecture or thought in clips up to 10 minutes, then review the text before saving.</p><div className="studio-controls">
    <button type="button" className="quiet-button is-primary" disabled={capture.busy||busy} onClick={capture.recording?capture.stop:capture.start}><UiIcon name={capture.recording?'stop':'mic'}/>{capture.recording?'Stop recording':'Start Recording'}</button>
    {(capture.recording||capture.busy)&&<button className="quiet-button" onClick={capture.cancel}>Cancel recording</button>}
    <button className="quiet-button" disabled={capture.recording||capture.busy||busy} onClick={()=>upload.current.click()}>Upload audio to transcribe</button><input ref={upload} hidden type="file" accept="audio/*" onChange={event=>{const file=event.target.files?.[0];event.target.value='';if(file){setBlob(file);capture.upload(file);}}}/>
  </div>{capture.recording&&<p role="status">Recording · {capture.seconds}s</p>}{capture.busy&&<p role="status">{capture.phase==='permission'?'Waiting for microphone permission…':'Transcribing your recording…'}</p>}</div>
    {(capture.error||error)&&<p role="alert" className="tool-error">{capture.error||error}</p>}
    {capture.audioUrl&&<audio controls src={capture.audioUrl} aria-label="Review recording"/>}
    {text&&<div className="simple-workspace-form"><label>Review transcript<textarea aria-label="Review transcript" rows={6} value={text} onChange={event=>{setText(event.target.value);setNotes('');}}/></label><div className="studio-controls"><button className="quiet-button" disabled={busy||!text.trim()} onClick={organize}>{busy?'Organizing…':'Organize as notes'}</button><button className="quiet-button is-primary" disabled={busy||!text.trim()} onClick={()=>onSave(notes||text,text,blob)}>{notes?'Save as Notes':'Save transcript'}</button></div>{notes&&<label>Review notes<textarea aria-label="Review notes" rows={8} value={notes} onChange={event=>setNotes(event.target.value)}/></label>}</div>}
    {capture.recordedBlob&&onSaveAudio&&<button className="quiet-button" onClick={()=>{const audio=capture.recordedBlob;onSaveAudio(new File([audio],'Recording.'+(/mp4/.test(audio.type)?'mp4':'webm'),{type:audio.type}));}}>Keep original audio</button>}
  </div>;
}
