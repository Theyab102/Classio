import {materialExcerpt} from './assistantRequest';
import {makeSourceBundle} from './sourceBundle';

export function linkedFilesFor(file,files) {
  return files.filter(other=>other.id!==file.id&&((file.linkedFileIds||[]).includes(other.id)||(other.linkedFileIds||[]).includes(file.id)));
}
export function setFileLinks(folders,fileId,selectedIds) {
  const files=folders.flatMap(folder=>folder.files||[]);
  if(!files.some(file=>file.id===fileId))return folders;
  const selected=new Set(selectedIds.filter(id=>id!==fileId&&files.some(file=>file.id===id)));
  return folders.map(folder=>({...folder,files:folder.files.map(file=>{
    const previous=file.linkedFileIds||[];
    const next=file.id===fileId?[...selected]:selected.has(file.id)?[...new Set([...previous,fileId])]:previous.filter(id=>id!==fileId);
    return next.length===previous.length&&next.every(id=>previous.includes(id))?file:{...file,linkedFileIds:next};
  })}));
}
export async function readLinkedStudySource(file,files,readOriginal,limit=16000,question='') {
  const current=files.find(item=>item.id===file.id);
  if(!current)throw new Error('This material is no longer available in this workspace.');
  const members=[current,...linkedFilesFor(current,files)];
  const sources=await Promise.all(members.map(async member=>{
    let original='';try{original=String(await readOriginal(member)||'').trim();}catch{}
    const saved=String(member.notes||member.transcript||'').trim();
    return {name:member.name,text:original||saved||'No readable text is available. Ask the learner to choose the original file on this device.',
      note:original?'Original material':saved?'Saved notes/transcript; original text is unavailable':'Original text unavailable; do not infer its contents from the filename'};
  }));
  if(!sources.some(source=>!source.note.startsWith('Original text unavailable')))return '';
  return materialExcerpt(makeSourceBundle(sources),question,limit);
}
