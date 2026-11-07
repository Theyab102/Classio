import { useEffect, useState } from "react";

const cache = new Map(), textureKeys=new Map();
function fingerprint(text){let a=2166136261,b=5381;for(let i=0;i<text.length;i++){const c=text.charCodeAt(i);a=Math.imul(a^c,16777619);b=Math.imul(b,33)^c;}return(a>>>0).toString(36)+(b>>>0).toString(36);}
function visualKey(look){return JSON.stringify(Object.fromEntries(Object.entries(look).map(([key,value])=>{if(!key.startsWith("custom")||!value?.texture)return[key,value];let hash=textureKeys.get(value.texture);if(!hash){hash=fingerprint(value.texture);if(textureKeys.size>=60)textureKeys.delete(textureKeys.keys().next().value);textureKeys.set(value.texture,hash);}return[key,{texture:hash,baseStyle:value.baseStyle,length:value.length,fringe:value.fringe,textureAmount:value.textureAmount}];})));}
let rendering = Promise.resolve();
function snapshot(key, character, options) {
  if (cache.has(key)) return Promise.resolve(cache.get(key));
  const work = rendering.then(() => new Promise(resolve => {
    // Yield between thumbnails so customizing a character never blocks scrolling.
    (window.requestIdleCallback || (callback => setTimeout(callback, 0)))(resolve, { timeout: 100 });
  })).then(async () => {
    if (cache.has(key)) return cache.get(key);
    const { renderCharacter } = await import("./avatarScene");
    const result = await renderCharacter(character, options);
    if (cache.size >= 256) cache.delete(cache.keys().next().value);
    cache.set(key, result); return result;
  });
  rendering = work.catch(() => {}); return work;
}

export default function FriendlyAvatar({ character = {}, size = 40, framing = "portrait", yaw = -.10 }) {
  const look = Object.fromEntries(Object.entries(character).filter(([field]) => !["name","designs","randomizerPreset","appStyle","previousAppStyle"].includes(field)));
  const key = JSON.stringify([visualKey(look), framing, yaw]);
  const [picture, setPicture] = useState(() => cache.get(key) || "");
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    if (typeof window.WebGL2RenderingContext === "undefined") { setFailed(true); return; }
    setFailed(false);
    snapshot(key, look, { framing, yaw, size: framing === "full" ? 384 : 192 }).then(url => { if (live) setPicture(url); }).catch(() => { if (live) setFailed(true); });
    return () => { live = false; };
  // The stable visual key excludes display names; typing a name does not rebuild meshes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return <img src={picture || (framing === "full" ? "/avatars/default-full-3d.png" : "/avatars/default-3d.png")} alt="Your Classio character" className="friendly-avatar" width={size} height={size} data-renderer={picture ? "three-dimensional" : failed ? "3d-poster" : "preparing"} data-hair-style={character.hairStyle || 0} style={{ objectFit: "contain", background: character.bg || "#e6f2fb" }} draggable="false" />;
}
