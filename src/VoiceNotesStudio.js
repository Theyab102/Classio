import {useEffect,useRef,useState} from 'react';
import VoiceRecorder from './VoiceRecorder';
import StudyMarkdown from './StudyMarkdown';
import {makeStudyId} from './studyTools';
function Recording({item,audioStore}){
  const store=useRef(audioStore);store.current=audioStore;const audioKey=item.audioKey;
  const[url,setUrl]=useState('');useEffect(()=>{let live=true,objectUrl='';if(audioKey)store.current.get(audioKey).then(blob=>{if(live&&blob){objectUrl=URL.createObjectURL(blob);setUrl(objectUrl);}}).catch(()=>{});return()=>{live=false;if(objectUrl)URL.revokeObjectURL(objectUrl);};},[audioKey]);
  return <article className="voice-note"><StudyMarkdown text={item.text||item.notes||item.transcript||''}/>{url&&<audio controls src={url} aria-label="Saved voice recording"/>}</article>;
}
export default function VoiceNotesStudio({file,onUpdate,audioStore}){
  const[error,setError]=useState(''),[version,setVersion]=useState(0);const recordings=file.voiceNotes||[];
  const saving=useRef(false),latest=useRef({file,onUpdate});latest.current={file,onUpdate};
  async function save(notes,transcript,audio){if(saving.current)return;saving.current=true;setError('');try{const id=makeStudyId(),audioKey='podcast:'+file.id+':recording:'+id;if(audio)await audioStore.save(audioKey,audio);const current=latest.current.file;latest.current.onUpdate({...current,voiceNotes:[...(current.voiceNotes||[]),{id,text:notes,transcript,...(audio?{audioKey}:{}),createdAt:Date.now()}],notes:[current.notes,notes].filter(Boolean).join('\n\n---\n\n')});setVersion(value=>value+1);}catch{setError('The recording could not be saved. Keep this page open and download the audio.');}finally{saving.current=false;}}
  return <section><VoiceRecorder key={version} onSave={save}/>{error&&<p className="tool-error" role="alert">{error}</p>}{recordings.length?recordings.map((item,index)=><div key={item.id||index}><Recording item={item} audioStore={audioStore}/><button className="quiet-button" onClick={()=>{if(item.audioKey)audioStore.remove(item.audioKey);onUpdate({...file,voiceNotes:recordings.filter((_,i)=>i!==index)});}}>Delete recording</button></div>):<div className="tool-empty"><h3>No recordings yet</h3><p>Your saved voice notes will appear here after you record.</p></div>}</section>;
}
