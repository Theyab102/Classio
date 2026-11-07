const subjects = {
  Physics:/\b(physics|projectile|kinematics|newton|gravity|acceleration|electromagnetism|thermodynamics|momentum)\b/i,
  Chemistry:/\b(chemistry|stoichiometry|molecule|chemical|periodic table|organic reaction|titration)\b/i,
  Biology:/\b(biology|photosynthesis|mitochondria|anatomy|respiration|genetics|cellular|ecology|reproductive)\b/i,
  Mathematics:/\b(mathematics|maths?|algebra|calculus|geometry|trigonometry|differentiation|integrals?)\b/i,
  History:/\b(history|historical|ancient|world war|civilization|renaissance)\b/i,
  Geography:/\b(geography|geographic|plate tectonic|climate zones|cartography)\b/i,
  'Computer Science':/\b(computer science|programming|algorithms|javascript|python|data structures|software engineering)\b/i,
  Economics:/\b(economics|macroeconomics|microeconomics|inflation|monetary policy)\b/i,
  Literature:/\b(literature|poetry|shakespeare|literary|novel analysis)\b/i,
};
export const SUBJECTS=[...Object.keys(subjects),'General'];
export function inferSubject(file) {
  if(file.manualSubject?.trim())return {label:file.manualSubject.slice(0,40),method:'manual',confident:true};
  for(const [source,method]of[[file.name||'','filename'],[[file.title,file.extractedText,file.searchText,file.transcript,file.notes].filter(Boolean).join(' ').replace(/<[^>]*>/g,' ').slice(0,10000),'content']]){
    const matches=Object.entries(subjects).filter(([,pattern])=>pattern.test(source.replace(/[_-]/g,' ')));
    if(matches.length===1)return {label:matches[0][0],method,confident:true};
  }
  if(file.detectedSubject&&file.detectedSubject!=='General'&&SUBJECTS.includes(file.detectedSubject))return{label:file.detectedSubject,method:file.subjectMethod||'ai',confident:true};
  return {label:'General',method:'unclassified',confident:false};
}
export function materialType(file={}) {
  const name=(file.name||'').toLowerCase(),type=(file.type||'').toLowerCase(),link=file.youtubeUrl||file.sourceUrl||file.url;
  if(link||type==='text/youtube'||type==='text/link'){
    try{const url=new URL(link);if(['youtube.com','www.youtube.com','m.youtube.com','youtu.be'].includes(url.hostname.toLowerCase()))return{label:'YouTube',icon:'youtube',key:'links'};return{label:'Website',icon:'globe',key:'links'};}catch{return{label:'Link',icon:'link',key:'links'};}
  }
  if(/\.pdf$/.test(name)||type.includes('pdf'))return{label:'PDF',icon:'source',key:'documents'};
  if(/\.pptx?$/.test(name)||/presentation|powerpoint/.test(type))return{label:'PowerPoint',icon:'slides',key:'documents'};
  if(/\.docx?$/.test(name)||/wordprocessing|msword/.test(type))return{label:'Word',icon:'notes',key:'documents'};
  if(type.startsWith('image/')||/\.(png|jpe?g|gif|webp|svg)$/.test(name))return{label:'Image',icon:'image',key:'images'};
  if(type.startsWith('audio/')||/\.(mp3|wav|m4a|ogg|webm)$/.test(name))return{label:'Audio',icon:'mic',key:'audio'};
  if(file._isBlank||file.isBlankDoc||file.id?.startsWith('blank_'))return{label:file.course?.units?.length?'Course & notes':'Note',icon:'notes',key:'notes'};
  return{label:'Document',icon:'source',key:'documents'};
}
export function materialMatches(file,folder,query){
  const terms=query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const text=[file.name,folder?.name,inferSubject(file).label,materialType(file).label,file.type,file.youtubeUrl,file.url,file.tags?.join(' '),file.searchText,file.extractedText,file.transcript,file.notes].filter(Boolean).join(' ').replace(/<[^>]*>/g,' ').toLowerCase();
  return terms.every(term=>text.includes(term));
}
