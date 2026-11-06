import {useEffect,useState} from 'react';
import {auth} from './firebase';
import {readPodcast} from './podcastAudio';
import {recallRecordingAudio} from './cloudWorkspace';
export default function SavedRecordingAudio({record}){
 const [url,setUrl]=useState(''),[error,setError]=useState('');
 useEffect(()=>{if(!record.audioCacheKey&&!record.audioCloud)return;let active=true,value='';(async()=>{let blob=await readPodcast(record.audioCacheKey).catch(()=>null);if(!blob&&auth.currentUser&&record.audioCloud)blob=await recallRecordingAudio(auth.currentUser.uid,record.audioCloud);if(blob&&active){value=URL.createObjectURL(blob);setUrl(value);}})().catch(()=>{if(active)setError('The original recording could not be loaded. Its notes are still available.');});return()=>{active=false;if(value)URL.revokeObjectURL(value);};},[record.audioCacheKey,record.audioCloud]);
 return url?<audio aria-label="Original voice recording" controls playsInline src={url} style={{width:'100%',marginTop:10}}/>:error?<p role="status">{error}</p>:null;
}
