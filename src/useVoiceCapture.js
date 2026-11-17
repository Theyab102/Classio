import {useEffect,useRef,useState} from 'react';
import {MAX_AUDIO_BYTES,sendStudyRequest} from './studyTools';
export function microphoneError(error){return error.name==='NotAllowedError'?'Microphone access was denied. Allow it in your browser’s site settings, then try again.':error.name==='NotFoundError'?'No microphone was found. Connect a microphone and try again.':error.name==='NotReadableError'?'The microphone is busy. Close other apps using it and try again.':error.message||'The microphone could not start.';}
export async function transcribeAudio(blob,signal){
  if(!blob.size)throw new Error('No audio was captured. Check your microphone and try again.');
  if(blob.size>MAX_AUDIO_BYTES)throw new Error('Use a recording under 2.8 MB. Record a shorter section or upload a smaller file.');
  const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(new Error('The recording could not be read.'));reader.readAsDataURL(blob);});
  const extension=/mp4|m4a/.test(blob.type)?'mp4':/ogg/.test(blob.type)?'ogg':/wav/.test(blob.type)?'wav':/mpeg|mp3/.test(blob.type)?'mp3':'webm';
  return sendStudyRequest({kind:'audio',audio:data,filename:'recording.'+extension},signal);
}
export default function useVoiceCapture(onTranscript,{maxSeconds=120}={}){
  const[phase,setPhase]=useState('idle'),[error,setError]=useState(''),[seconds,setSeconds]=useState(0),[audioUrl,setAudioUrl]=useState(''),[recordedBlob,setRecordedBlob]=useState(null);
  const latest=useRef(onTranscript),run=useRef(null),url=useRef('');latest.current=onTranscript;
  const release=job=>{clearInterval(job?.timer);clearTimeout(job?.limit);job?.stream?.getTracks().forEach(track=>track.stop());};
  const cancel=()=>{const job=run.current;run.current=null;job?.controller.abort();if(job?.recorder?.state==='recording')job.recorder.stop();release(job);setPhase('idle');};
  useEffect(()=>()=>{const job=run.current;run.current=null;job?.controller.abort();if(job?.recorder?.state==='recording')job.recorder.stop();release(job);if(url.current)URL.revokeObjectURL(url.current);},[]);
  async function transcribe(blob,job){
    if(run.current!==job)return;setPhase('transcribing');
    if(url.current)URL.revokeObjectURL(url.current);url.current=URL.createObjectURL(blob);setAudioUrl(url.current);setRecordedBlob(blob);
    try{const text=await transcribeAudio(blob,job.controller.signal);if(run.current!==job)return;if(!text.trim())throw new Error('No speech was recognized. Try again or type your message.');latest.current(text.trim(),blob);setPhase('idle');}
    catch(failure){if(run.current===job&&!job.controller.signal.aborted){setError(microphoneError(failure));setPhase('idle');}}
  }
  async function start(){
    cancel();setError('');setSeconds(0);
    if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder){setError('Microphone recording needs HTTPS and a supported browser. You can type or upload audio instead.');return;}
    const job={controller:new AbortController()};run.current=job;setPhase('permission');
    try{
      const stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});job.stream=stream;if(run.current!==job){release(job);return;}
      const type=['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus','audio/webm'].find(value=>MediaRecorder.isTypeSupported?.(value)),recorder=new MediaRecorder(stream,{...(type?{mimeType:type}:{}),audioBitsPerSecond:32000}),chunks=[];let size=0;job.recorder=recorder;
      recorder.ondataavailable=event=>{if(event.data.size){chunks.push(event.data);size+=event.data.size;if(size>MAX_AUDIO_BYTES&&recorder.state==='recording')recorder.stop();}};
      recorder.onerror=()=>{if(run.current===job){cancel();setError('Recording failed. Check your microphone and try again.');}};
      recorder.onstop=()=>{release(job);if(run.current===job&&!job.controller.signal.aborted)transcribe(new Blob(chunks,{type:recorder.mimeType||type||'audio/webm'}),job);};
      recorder.start(250);setPhase('recording');job.timer=setInterval(()=>setSeconds(value=>value+1),1000);job.limit=setTimeout(()=>{if(recorder.state==='recording')recorder.stop();},maxSeconds*1000);
    }catch(failure){release(job);if(run.current===job){setError(microphoneError(failure));setPhase('idle');}}
  }
  async function upload(blob){cancel();setError('');const job={controller:new AbortController()};run.current=job;await transcribe(blob,job);}
  return{phase,error,seconds,audioUrl,recordedBlob,start,stop:()=>{if(run.current?.recorder?.state==='recording')run.current.recorder.stop();},cancel,upload,recording:phase==='recording',busy:phase==='permission'||phase==='transcribing'};
}
