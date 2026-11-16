import assets from './localScriptManifest.json';
const pending=new Map();
export function loadLocalScript(name){
 const asset=assets.find(item=>item.name===name);if(!asset)return Promise.reject(new Error('This document tool is unavailable.'));
 if(pending.has(name))return pending.get(name);
 const work=new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='/vendor/'+asset.file;script.integrity='sha384-'+asset.sha384;script.crossOrigin='anonymous';script.referrerPolicy='no-referrer';script.onload=resolve;script.onerror=()=>{script.remove();pending.delete(name);reject(new Error('The document tool could not load. Try again.'));};document.head.append(script);});
 pending.set(name,work);return work;
}
