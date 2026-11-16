import {makeStudyId} from './studyTools';

export const SLIDE_TYPES=[['title','Title','Open with a clear idea'],['content','Content','Explain the key points'],['image','Image','Let a figure lead'],['two-col','Comparison','Compare two ideas'],['timeline','Timeline','Show a sequence'],['summary','Summary','Bring the ideas together'],['table','Table','Present structured facts'],['quote','Quote','Highlight a source']];
export const DECK_THEMES=[
 {id:'modern',label:'Midnight',bg:'#1e1b4b',accent:'#a5b4fc',text:'#f8fafc',sub:'#d5d5ec'},
 {id:'ocean',label:'Ocean',bg:'#0c4a6e',accent:'#7dd3fc',text:'#f0f9ff',sub:'#c7e9fa'},
 {id:'minimal',label:'Paper',bg:'#f8f9fa',accent:'#245f8c',text:'#182b3d',sub:'#526477'},
 {id:'forest',label:'Forest',bg:'#14532d',accent:'#86efac',text:'#f0fdf4',sub:'#d0f2dc'},
 {id:'sunset',label:'Terracotta',bg:'#431407',accent:'#fdba74',text:'#fff7ed',sub:'#fed7aa'},
 {id:'dark',label:'Slate',bg:'#0f172a',accent:'#93c5fd',text:'#f1f5f9',sub:'#cbd5e1'},
 {id:'rose',label:'Rose',bg:'#390b26',accent:'#fda4af',text:'#fff1f2',sub:'#fecdd3'},
 {id:'teal',label:'Teal',bg:'#022c22',accent:'#5eead4',text:'#f0fdfa',sub:'#b0efe1'},
 {id:'slate',label:'Ink',bg:'#0f172a',accent:'#7dd3fc',text:'#f1f5f9',sub:'#cbd5e1'},
 {id:'amber',label:'Amber',bg:'#1c0a00',accent:'#fbbf24',text:'#fffbeb',sub:'#fde68a'},
];
export function createSlide(type='content'){
 const base={id:makeStudyId(),type,title:type==='title'?'Your next idea':type==='timeline'?'How it unfolds':type==='summary'?'The takeaway':'New slide'};
 if(type==='title')return {...base,subtitle:'A presentation made in Classio'};
 if(type==='quote')return {...base,title:'',quote:'A thought worth sharing',author:''};
 if(type==='two-col')return {...base,left:{heading:'First idea',bullets:['Key point']},right:{heading:'Second idea',bullets:['Key point']}};
 if(type==='table')return {...base,table:[['Topic','Detail'],['','']]};
 return {...base,bullets:type==='image'?[]:['First point','Second point']};
}
export const withSlideIds=slides=>(slides||[]).map(slide=>slide.id?slide:{...slide,id:makeStudyId()});
export const updateSlideById=(slides,id,patch)=>slides.map(slide=>slide.id===id?{...slide,...patch}:slide);
export const copySlide=slide=>({...JSON.parse(JSON.stringify(slide)),id:makeStudyId()});
export function imageCrop(value={}){
 value=value||{};
 const bound=(n,min,max,fallback)=>Number.isFinite(Number(n))?Math.max(min,Math.min(max,Number(n))):fallback;
 const x=bound(value.x,0,.95,0),y=bound(value.y,0,.95,0);
 return {x,y,w:Math.round(bound(value.w,.05,1-x,1-x)*10000)/10000,h:Math.round(bound(value.h,.05,1-y,1-y)*10000)/10000};
}
export async function preparedSlideImage(slide){
 const {powerpointImage}=await import('./presentationImages');
 const crop=imageCrop(slide.imageCrop);
 if(crop.x===0&&crop.y===0&&crop.w===1&&crop.h===1)return powerpointImage(slide.image);
 const img=new Image();img.crossOrigin='anonymous';img.referrerPolicy='no-referrer';img.src=slide.image;await img.decode();
 const canvas=document.createElement('canvas');canvas.width=Math.max(1,Math.round(img.naturalWidth*crop.w));canvas.height=Math.max(1,Math.round(img.naturalHeight*crop.h));
 canvas.getContext('2d').drawImage(img,img.naturalWidth*crop.x,img.naturalHeight*crop.y,img.naturalWidth*crop.w,img.naturalHeight*crop.h,0,0,canvas.width,canvas.height);
 return canvas.toDataURL('image/png');
}
