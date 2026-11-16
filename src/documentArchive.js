import JSZip from 'jszip';
export function validateArchive(zip,{maxPartBytes=20*1024*1024,maxTotalBytes=100*1024*1024}={}){
 const entries=Object.values(zip.files),budget={readBytes:0};let expanded=0;
 if(entries.length>4000)throw new Error('This document contains too many parts. Use a smaller copy.');
 for(const entry of entries){
  const size=entry._data?.uncompressedSize||0;expanded+=size;
  if(size>maxPartBytes||expanded>maxTotalBytes)throw new Error('This document expands beyond the reader’s size limit. Use a smaller copy.');
  if(entry.dir||!entry.internalStream)continue;
  // ZIP size declarations are untrusted. Bound actual emitted data too, before
  // JSZip can accumulate a dishonest entry into one very large string/buffer.
  entry.async=(type,onUpdate)=>new Promise((resolve,reject)=>{
   const stream=entry.internalStream(type);let bytes=0,failed=false;
   stream.on('data',chunk=>{
    if(failed)return;
    const length=typeof chunk==='string'?chunk.length*(type==='string'||type==='text'?2:1):chunk.byteLength||chunk.length;
    bytes+=length;budget.readBytes+=length;
    if(bytes>maxPartBytes||budget.readBytes>maxTotalBytes){failed=true;stream.pause();reject(new Error('This document expands beyond the reader’s size limit. Use a smaller copy.'));}
   });
   stream.accumulate(onUpdate).then(value=>{if(!failed)resolve(value);},error=>{if(!failed)reject(error);});
  });
 }
 return zip;
}
export async function openDocumentArchive(file){
 if(file.size>25*1024*1024)throw new Error('Use an Office document under 25 MB.');
 return validateArchive(await JSZip.loadAsync(await file.arrayBuffer()));
}
export async function verifyDocumentArchive(file){
 const zip=await openDocumentArchive(file);
 for(const entry of Object.values(zip.files))if(!entry.dir)await entry.async('uint8array');
}
