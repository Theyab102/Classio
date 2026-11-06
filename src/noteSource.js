import { TTS_SPACE_URL } from './studyTools';
import { youtubeVideo, normalizeStudyLink } from './youtube';

async function readLink(url, signal, json = false) {
  const controller = new AbortController();
  const cancel = () => controller.abort();
  signal?.addEventListener('abort', cancel, { once:true });
  if (signal?.aborted) cancel();
  const timer = setTimeout(cancel,12000);
  try {
    const response = await fetch(url,{signal:controller.signal,credentials:'omit',referrerPolicy:'no-referrer'});
    if(!response.ok) throw new Error('Source unavailable');
    return json ? await response.json() : await response.text();
  } finally {clearTimeout(timer);signal?.removeEventListener('abort',cancel);}
}
export async function resolveNoteSource({ file, pastedText = '', existing = '', getFileText, signal }) {
  if(pastedText.trim()) return {text:pastedText.trim(),label:'supplied source text'};
  if(file.transcript?.trim()) return {text:file.transcript,label:'provided transcript'};
  if(file.sourceText?.trim()) return {text:file.sourceText,label:'saved source text'};
  const extracted = await getFileText?.();
  if(typeof extracted==='string' && extracted.trim()) {
    if(extracted.startsWith('[This is an image file:')) throw new Error('This image has no extracted text. Paste its text in Generation options before generating notes.');
    return {text:extracted,label:'uploaded study file'};
  }
  if(file.youtubeUrl || file.sourceUrl) {
    const url=normalizeStudyLink(file.youtubeUrl || file.sourceUrl);
    const video=youtubeVideo(url);
    if(video) {
      try {
        if(!TTS_SPACE_URL) throw new Error();
        const data=await readLink(`${TTS_SPACE_URL}/transcript/${video.id}`,signal,true);
        if(data.success && typeof data.transcript==='string' && data.transcript.trim().length>20) return {text:data.transcript,label:'available video transcript'};
      } catch {}
      throw new Error('A video transcript is not available. Paste it in Generation options; the video has not been analyzed.');
    }
    try {
      const html=await readLink(url,signal);
      const document=new DOMParser().parseFromString(html,'text/html');
      document.querySelectorAll('script,style,nav,header,footer,noscript').forEach(node=>node.remove());
      const text=(document.body.textContent || '').replace(/\s+/g,' ').trim();
      if(text.length>80) return {text:text.slice(0,20000),label:'readable linked webpage'};
    } catch {}
    throw new Error('This link could not be read. Paste the source text in Generation options to make notes from it.');
  }
  return existing.trim() ? {text:existing,label:'your existing notes'} : {text:'',label:''};
}
