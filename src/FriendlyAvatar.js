import { useEffect, useRef, useState } from "react";

const cache = new Map(), textureKeys=new Map();
function fingerprint(text){let a=2166136261,b=5381;for(let i=0;i<text.length;i++){const c=text.charCodeAt(i);a=Math.imul(a^c,16777619);b=Math.imul(b,33)^c;}return(a>>>0).toString(36)+(b>>>0).toString(36);}
function visualKey(look){return JSON.stringify(Object.fromEntries(Object.entries(look).map(([key,value])=>{if(!key.startsWith("custom")||!value?.texture)return[key,value];let hash=textureKeys.get(value.texture);if(!hash){hash=fingerprint(value.texture);if(textureKeys.size>=60)textureKeys.delete(textureKeys.keys().next().value);textureKeys.set(value.texture,hash);}return[key,{texture:hash,baseStyle:value.baseStyle,length:value.length,fringe:value.fringe,textureAmount:value.textureAmount}];})));}
const pending=new Map(),queue=[];let running=false;
async function drain(){
  if(running)return;running=true;
  try{while(queue.length){
    await new Promise(resolve=>requestAnimationFrame(resolve));
    queue.sort((a,b)=>b.priority-a.priority);const job=queue.shift();
    if(!job.listeners.some(active=>active())){pending.delete(job.key);job.resolve(null);continue;}
    try{const{renderCharacter}=await import("./avatarScene");const result=await renderCharacter(job.character,job.options);if(cache.size>=256)cache.delete(cache.keys().next().value);cache.set(job.key,result);job.resolve(result);}catch(error){job.reject(error);}finally{pending.delete(job.key);}
  }}finally{running=false;}
}
function snapshot(key,character,options,active,priority){
  if(cache.has(key))return Promise.resolve(cache.get(key));
  if(pending.has(key)){const job=pending.get(key);job.listeners.push(active);job.priority=Math.max(job.priority,priority);return job.promise;}
  let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;}),job={key,character,options,listeners:[active],priority,promise,resolve,reject};pending.set(key,job);queue.push(job);drain();return promise;
}

export default function FriendlyAvatar({ character = {}, size = 40, framing = "portrait", yaw = -.10, lazy = false }) {
  const look = Object.fromEntries(Object.entries(character).filter(([field]) => !["name","designs","randomizerPreset","appStyle","previousAppStyle"].includes(field)));
  const key = JSON.stringify([visualKey(look), framing, yaw]);
  const element=useRef(null),[visible,setVisible]=useState(!lazy||typeof IntersectionObserver==="undefined");
  useEffect(()=>{if(!lazy||typeof IntersectionObserver==="undefined")return;const observer=new IntersectionObserver(entries=>setVisible(entries[0].isIntersecting),{rootMargin:"80px"});if(element.current)observer.observe(element.current);return()=>observer.disconnect();},[lazy]);
  const [picture, setPicture] = useState(() => cache.get(key) || "");
  const [pictureKey,setPictureKey]=useState(()=>cache.has(key)?key:"");
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;if(!visible)return;
    if (typeof window.WebGL2RenderingContext === "undefined") { setFailed(true); return; }
    setFailed(false);
    snapshot(key, look, { framing, yaw, size: framing === "full" ? 384 : 192 },()=>live,size>=120?1:0).then(url => { if (live&&url) {setPicture(url);setPictureKey(key);} }).catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  // The stable visual key excludes display names; typing a name does not rebuild meshes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key,visible]);
  return <img ref={element} src={picture || (framing === "full" ? "/avatars/default-full-3d.png" : "/avatars/default-3d.png")} alt="Your Classio character" className="friendly-avatar" width={size} height={size} aria-busy={!failed&&pictureKey!==key} data-renderer={picture && pictureKey===key ? "three-dimensional" : failed ? "3d-poster" : "preparing"} data-hair-style={character.hairStyle || 0} style={{ objectFit: "contain", background: character.bg || "#e6f2fb" }} draggable="false" />;
}
