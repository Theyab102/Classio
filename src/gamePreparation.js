import {startStudyWork} from './studyWork';
import {studySource} from './studyContext';
import {generateStudySet} from './checkpointGenerators';
import {readGenerationJob,appliedGenerationPatch} from './generationJobs';
import {parseFlashcards} from './FlashcardStudio';

// Games and Flashcards share one recoverable, account-owned card-generation job.
export function prepareGameCards({file,getFileText,resume,automatic=false,fallback}){
  return startStudyWork({file,kind:'flashcards',fallback,work:async context=>{
    const saved=resume||await readGenerationJob(file,'flashcards');context.check();
    const unfinished=saved&&saved.status!=='completed'?saved:null;
    if(automatic&&unfinished&&['paused','failed'].includes(unfinished.status))throw new DOMException('Preparation paused','AbortError');
    const source=unfinished?.input.source??await studySource(file,getFileText);context.check();
    if(!source.trim())throw new Error('Add notes or an original document so the games can use your material.');
    const input=unfinished?.input||{source,topic:file.name,count:10,replace:false,avoid:(file.studyCards||[]).map(card=>card.question)};
    let checkpoint;
    const next=await generateStudySet({fileId:file.id,kind:'flashcards',input,resume:unfinished,parse:parseFlashcards,signal:context.signal,onProgress:context.progress,onCheckpoint:job=>{checkpoint=job;context.checkpoint(job);}});
    context.finishing();context.commit(current=>{
      if(current.generationJobs?.flashcards?.id===checkpoint.id&&current.generationJobs.flashcards.status==='completed')return current;
      const normalize=text=>text.normalize('NFKC').replace(/\s+/g,' ').trim().toLowerCase(),existing=current.studyCards||[],seen=new Set(existing.map(card=>normalize(card.question)));
      const additions=next.filter(card=>{const text=normalize(card.question);if(seen.has(text))return false;seen.add(text);return true;});
      return {...current,...appliedGenerationPatch(current,'flashcards',checkpoint),studyCards:input.replace?next:[...existing,...additions],...(input.replace?{cardProgress:{}}:{})};
    });return next;
  }});
}
