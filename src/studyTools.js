import {aiOwner} from './aiAccess';
import {acquireStudySlot,studyCapacityDeadline,deferStudyCapacity,waitForStudyCapacity,retryDelay} from './studyQueue';
import {aiMode} from './aiPreferences';
import {authorizedFetch} from './aiAccess';
import {studyStyleInstruction} from './studyContext';
async function sendStudyOnce(payload, signal, owner) {
  if(signal?.aborted||aiOwner()!==owner)throw new DOMException("Cancelled","AbortError");
  const mode=aiMode();if(mode==="browser-local"){const{generateLocal}=await import("./localAi");if(signal?.aborted||aiOwner()!==owner)throw new DOMException("Cancelled","AbortError");return generateLocal(payload,signal);}
  const controller = new AbortController();
  const cancel = () => controller.abort();
  if (signal?.aborted) cancel();
  signal?.addEventListener("abort", cancel, { once: true });
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, 75000);
  try {
    const response = await authorizedFetch("/api/study", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({...payload,provider:payload.provider||mode,messages:(payload.messages || []).map((message,index)=>index===0&&message.role==="system"&&!(["style","classification"].includes(payload.task))?{...message,content:message.content+"\n"+studyStyleInstruction()}:message)}), signal: controller.signal,
    },owner);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || typeof data.text !== "string" || !data.text.trim()) {
      const error = new Error(data.error || "The AI service could not finish this request. Your work is still open.");
      error.code = data.code || "STUDY_UNAVAILABLE"; error.status = response.status;
      error.retryAfter = retryDelay(response.headers.get("retry-after"));
      throw error;
    }
    return data.text;
  } catch (error) {
    if (timedOut) throw new Error("The AI service took too long. Try a shorter request or try again shortly.");
    if (!signal?.aborted && error instanceof TypeError) throw new Error("Could not reach the study service. Check your connection and try again.");
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}


export async function sendStudyRequest(payload,signal) {
  const owner=aiOwner(),modality=payload.audio||payload.kind==='audio'?'audio':payload.vision||(payload.messages||[]).some(message=>Array.isArray(message.content)&&message.content.some(part=>part.type==='image_url'))?'vision':'text',quota=modality==='audio'?'audio':'study';
  const check=()=>{if(signal?.aborted||aiOwner()!==owner)throw new DOMException('Generation cancelled','AbortError');};
  const deadline=()=>aiMode()==='browser-local'?0:Math.max(studyCapacityDeadline(owner,modality),studyCapacityDeadline(owner,'quota:'+quota),studyCapacityDeadline(owner,'session'));
  for(;;){
    check();const until=deadline();
    if(until>Date.now())await waitForStudyCapacity(until,signal,owner);
    const release=await acquireStudySlot(owner,signal,quota);
    try{
      check();if(deadline()>Date.now())continue;
      const text=await sendStudyOnce(payload,signal,owner);check();return text;
    }catch(error){
      if(error.status!==429||signal?.aborted||aiOwner()!==owner)throw error;
      deferStudyCapacity(owner,error.retryAfter||30,error.capacityScope==='session'?'session':error.code==='CLIENT_RATE_LIMIT'?'quota:'+quota:modality);
    }finally{release();}
  }
}

export const MAX_AUDIO_BYTES = 2_800_000;

export function mergeSessionFolders(accountFolders, sessionFolders) {
  const folders = new Map(accountFolders.map(folder => [folder.id, folder]));
  for (const folder of sessionFolders) {
    const saved = folders.get(folder.id);
    const files = new Map((saved?.files || []).map(file => [file.id, file]));
    for (const file of folder.files) files.set(file.id, file);
    folders.set(folder.id, { ...(saved || folder), files: [...files.values()] });
  }
  return [...folders.values()];
}

const reviewIntervals = [1, 3, 7, 14, 30];

export function scheduleCardReview(previous, remembered, now = Date.now()) {
  if (!remembered) return { level: 0, due: now + 10 * 60 * 1000 };
  const level = Math.min((previous?.level || 0) + 1, reviewIntervals.length);
  return { level, due: now + reviewIntervals[level - 1] * 24 * 60 * 60 * 1000 };
}

export const TTS_SPACE_URL = (process.env.REACT_APP_CLASSIO_TTS_URL ?? "https://theyab102-classio-tts.hf.space").replace(/\/$/, "");

export function downloadStudyFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function makeStudyId() {
  return window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
