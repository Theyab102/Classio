import {useEffect,useRef,useState} from 'react';
import StudioDialog from './StudioDialog';
import SlideCanvasEditor from './SlideCanvasEditor';
import PresentationImageEditor from './PresentationImageEditor';
import {PlacementCards} from './PresentationOptions';
import {SLIDE_TYPES,imageCrop} from './presentationModel';
import {normalizeSlideTable} from './DeckDesign';
import {preparePresentationImage} from './presentationImages';
import {getSourceFigures,resolvePresentationFigure} from './sourceFigures';
import {acknowledgeStudyImage,searchStudyImages} from './documentMedia';
import {requestImage} from './imageService';
import useActionHistory from './useActionHistory';
import UiIcon from './UiIcon';

export default function SlideEditor({slide,index,total,theme,layout,design,fileObj,onApply,onClose,onMove,onDuplicate,onDelete,onGenerateImage}){
 const scope=useRef(null),request=useRef(null),version=useRef(0),upload=useRef(null),titleField=useRef(null),bodyField=useRef(null);
 const [draft,setDraft,history]=useActionHistory(()=>JSON.parse(JSON.stringify(slide)),scope,slide.id);
 const [section,setSection]=useState('content'),[error,setError]=useState(''),[notice,setNotice]=useState(''),[busy,setBusy]=useState(''),[results,setResults]=useState([]),[query,setQuery]=useState(''),[prompt,setPrompt]=useState(''),[imageStyle,setImageStyle]=useState('Educational diagram'),[cutout,setCutout]=useState(false),[saving,setSaving]=useState(false);
 const dirty=JSON.stringify(draft)!==JSON.stringify(slide),latest=useRef(draft);latest.current=draft;
 const patch=fields=>{version.current++;setDraft(current=>({...current,...fields}));};
 const close=()=>{if(dirty&&!window.confirm('Discard your unsaved slide changes?'))return;request.current?.abort();onClose();};
 useEffect(()=>()=>request.current?.abort(),[]);
 useEffect(()=>{const warn=event=>{if(dirty){event.preventDefault();event.returnValue='';}};window.addEventListener('beforeunload',warn);return()=>window.removeEventListener('beforeunload',warn);},[dirty]);
 async function task(label,work){
   request.current?.abort();const controller=new AbortController();request.current=controller;const revision=version.current;setBusy(label);setError('');
   try{const value=await work(controller.signal);if(controller.signal.aborted)return;
     if(revision!==version.current){setNotice('Your slide changed while this request was running. Your edits are kept; try again.');return;}
     return value;
   }catch(e){if(!controller.signal.aborted)setError(e.message||'This action could not finish. Your slide is kept.');}
   finally{if(request.current===controller){request.current=null;setBusy('');}}
 }
 async function find(source){
   const found=await task('Finding images',async signal=>source==='file'?Promise.all((await getSourceFigures(fileObj)).slice(0,24).map(item=>resolvePresentationFigure(item,fileObj))):searchStudyImages(query||draft.imageSearch||draft.title,signal));
   if(found){setResults(found.filter(Boolean));if(!found.length)setNotice('No matching images. Try a more specific search or upload your own.');}
 }
 async function choose(image){
   const ready=await task('Preparing image',async signal=>{await acknowledgeStudyImage(image,signal);return image;});
   if(ready){patch({image:ready.src,imageOriginal:null,imageAlt:ready.title,imageCredit:ready.credit||'',imageSource:ready.source||'',imageCrop:null,...(ready.width&&ready.height?{imageAspect:ready.width/ready.height}:{})});setResults([]);}
 }
 async function generate(){
   const image=await task('Generating illustration',async signal=>{
     const description=`${imageStyle}. ${prompt.trim()||[draft.title,...(draft.bullets||[]),draft.subtitle].filter(Boolean).join('. ')}. Preserve scientific meaning; no decorative logos.`;
     const capabilities=await fetch('/api/images',{signal}).then(response=>response.ok?response.json():{}).catch(()=>({}));
     if(signal.aborted)throw new DOMException('Cancelled','AbortError');
     if(capabilities.imageEditing)return requestImage({action:'generate',prompt:description},signal);
     if(onGenerateImage){const style={'Educational diagram':'diagram','Scientific illustration':'illustration','Infographic':'infographic','Photograph':'realistic','Minimal illustration':'minimal'}[imageStyle];return(await onGenerateImage(description,style,signal)).url;}
     throw new Error('No image-generation service is configured. Upload a figure or search for one instead.');
   });
   if(image)patch({image,imageOriginal:null,imageAlt:prompt.trim()||draft.title,imageCredit:'AI-generated illustration',imageSource:'',imageCrop:null});
 }
 async function save(){
   const {tableInput,...next}=draft;
   if(tableInput!==undefined)next.table=normalizeSlideTable(tableInput.split('\n').filter(row=>row.trim()).map(row=>row.split('|')));
   if(next.type==='table'&&!normalizeSlideTable(next.table)){setError('Add at least two rows and two columns to the table.');return;}
   setSaving(true);setError('');try{await onApply(next);onClose();}catch{setError('Changes could not be applied. Your draft is still here; try again.');}finally{setSaving(false);}
 }
 function editText(key){setSection('content');setTimeout(()=>{(key==='title'?titleField:bodyField).current?.focus();},0);}
 const crop=imageCrop(draft.imageCrop),typeName=SLIDE_TYPES.find(([id])=>id===draft.type)?.[1]||'Slide';
 return <StudioDialog title={`Slide ${index+1} · ${typeName}`} wide onClose={close}><div ref={scope} className="slide-editor" aria-label="Slide editor">
  <div className="slide-editor-toolbar" role="toolbar" aria-label="Edit slide actions"><span className="draft-state" role="status">{dirty?'Unsaved changes':'No unsaved changes'}</span>
   <button className="quiet-button" disabled={index===0||dirty} onClick={()=>onMove(-1)} aria-label="Move edited slide up"><UiIcon name="back"/>Move up</button><button className="quiet-button" disabled={index===total-1||dirty} onClick={()=>onMove(1)}>Move down<UiIcon name="arrow"/></button>
   <button className="quiet-button" disabled={dirty} onClick={onDuplicate}>Duplicate</button><button className="quiet-button" disabled={!history.canUndo} onClick={()=>{version.current++;history.undo();}}><UiIcon name="undo"/>Undo</button><button className="quiet-button" disabled={!history.canRedo} onClick={()=>{version.current++;history.redo();}}><UiIcon name="redo"/>Redo</button>
   <button className="quiet-button is-danger" onClick={()=>{if(window.confirm('Delete this slide?'))onDelete();}}><UiIcon name="trash"/>Delete</button>
  </div>
  <div className="slide-editor-grid"><div className="slide-editor-tools"><nav className="slide-tool-tabs" aria-label="Slide editing tools">{[['content','Text & content'],['images','Images & media'],['layout','Layout & design'],['ai','AI tools'],['background','Background & styling']].map(([id,label])=><button className="quiet-button" key={id} aria-pressed={section===id} onClick={()=>setSection(id)}>{label}</button>)}</nav>
   {section==='content'&&<div className="slide-tool-fields"><label>Slide type<select aria-label="Slide type" value={draft.type} onChange={e=>patch({type:e.target.value})}>{SLIDE_TYPES.map(([id,name])=><option key={id} value={id}>{name}</option>)}</select></label>
    {draft.type!=='quote'&&<label>Title<textarea ref={titleField} aria-label="Slide title" rows={2} maxLength={150} value={draft.title||''} onChange={e=>patch({title:e.target.value})}/></label>}
    {draft.type==='title'&&<label>Subtitle<textarea ref={bodyField} rows={3} value={draft.subtitle||''} onChange={e=>patch({subtitle:e.target.value})}/></label>}
    {['content','summary','timeline','image'].includes(draft.type)&&<label>{draft.type==='timeline'?'Steps (one per line)':'Key points (one per line)'}<textarea ref={bodyField} aria-label="Slide key points" rows={7} value={(draft.bullets||[]).join('\n')} onChange={e=>patch({bullets:e.target.value.split('\n')})}/></label>}
    {draft.type==='two-col'&&['left','right'].map(side=><fieldset key={side}><legend>{side==='left'?'First idea':'Second idea'}</legend><label>Heading<input value={draft[side]?.heading||''} onChange={e=>patch({[side]:{...draft[side],heading:e.target.value}})}/></label><label>Points<textarea rows={4} value={(draft[side]?.bullets||[]).join('\n')} onChange={e=>patch({[side]:{...draft[side],bullets:e.target.value.split('\n')}})}/></label></fieldset>)}
    {draft.type==='table'&&<label>Table rows · separate cells with |<textarea aria-label="Slide table rows" rows={7} value={draft.tableInput??(draft.table||[['Column','Value'],['','']]).map(row=>row.join(' | ')).join('\n')} onChange={e=>{const value=e.target.value;patch({tableInput:value,table:normalizeSlideTable(value.split('\n').filter(row=>row.trim()).map(row=>row.split('|')))})}}/></label>}
    {draft.type==='quote'&&<><label>Quote<textarea ref={bodyField} rows={5} value={draft.quote||''} onChange={e=>patch({quote:e.target.value})}/></label><label>Attribution<input value={draft.author||''} onChange={e=>patch({author:e.target.value})}/></label></>}
    <label>Speaker notes<textarea rows={4} value={draft.note||''} onChange={e=>patch({note:e.target.value})}/></label>
   </div>}
   {section==='images'&&<div className="slide-tool-fields">{draft.image&&<figure className="editor-image-preview"><img src={draft.image} alt={draft.imageAlt||'Slide image'} referrerPolicy="no-referrer"/><figcaption>{draft.imageCredit||'Current slide image'}</figcaption></figure>}
    <div className="studio-controls"><button className="quiet-button" disabled={!!busy} onClick={()=>upload.current.click()}><UiIcon name="upload"/>Upload image</button>{draft.image&&<button className="quiet-button" onClick={()=>patch({image:'',imageOriginal:null,imageCrop:null,geometry:{...draft.geometry,image:undefined}})}>Remove image</button>}</div>
    <input hidden type="file" ref={upload} accept="image/png,image/jpeg,image/webp,image/gif" onChange={async e=>{const f=e.target.files?.[0];e.target.value='';if(!f)return;const value=await task('Preparing upload',()=>preparePresentationImage(f));if(value)patch({image:value.src,imageAspect:value.width/value.height,imageOriginal:null,imageAlt:f.name,imageCredit:'',imageSource:'',imageCrop:null});}}/>
    <label>Search for a figure<input aria-label="Slide image search" value={query} onChange={e=>setQuery(e.target.value)} placeholder={draft.title||'e.g. cell division diagram'}/></label><div className="studio-controls"><button className="quiet-button" disabled={!!busy} onClick={()=>find('web')}>Search images</button>{fileObj&&<button className="quiet-button" disabled={!!busy} onClick={()=>find('file')}>Choose from source file</button>}</div>
    <div className="slide-image-results">{results.map((image,i)=><button key={image.src+i} onClick={()=>choose(image)} disabled={!!busy}><img src={image.src} alt={image.title} referrerPolicy="no-referrer"/><span>{image.title}</span><small>{image.credit}</small></button>)}</div>
    {draft.image&&<><button className="quiet-button" onClick={()=>setCutout(true)}>Remove Background</button>{draft.imageOriginal&&<button className="quiet-button" onClick={()=>patch({image:draft.imageOriginal,imageCrop:null})}>Restore original image</button>}<label>Alternative text<input value={draft.imageAlt||''} onChange={e=>patch({imageAlt:e.target.value})}/></label>
     <fieldset><legend>Crop</legend>{[['x','Left'],['y','Top'],['w','Width'],['h','Height']].map(([key,label])=><label key={key}>{label}<input aria-label={'Crop '+label.toLowerCase()} type="range" min={key==='w'||key==='h'?5:0} max={100} value={Math.round(crop[key]*100)} onChange={e=>patch({imageCrop:imageCrop({...crop,[key]:Number(e.target.value)/100})})}/></label>)}<button className="text-button" onClick={()=>patch({imageCrop:null})}>Reset crop</button></fieldset></>}
   </div>}
   {section==='layout'&&<div className="slide-tool-fields"><p>The presentation uses the {layout} layout. Change the deck layout outside the editor.</p><PlacementCards value={draft.imageMode==='background'?'background':draft.imagePlacement} onChange={value=>patch({imageMode:value==='background'?'background':undefined,imagePlacement:value==='background'?'auto':value,geometry:undefined})}/><label className="check-label"><input type="checkbox" checked={draft.lockAspect!==false} onChange={e=>patch({lockAspect:e.target.checked})}/>Preserve image aspect ratio when resizing</label><button className="quiet-button" onClick={()=>patch({geometry:undefined,imagePlacement:'auto'})}>Reset element positions</button><p>Select text or an image in the preview, then drag or use arrow keys. Shift moves farther. Use the corner handle to resize.</p></div>}
   {section==='ai'&&<div className="slide-tool-fields"><label>Describe the illustration<textarea rows={4} maxLength={3000} value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="Leave blank to use this slide’s content"/></label><label>Visual style<select value={imageStyle} onChange={e=>setImageStyle(e.target.value)}>{['Educational diagram','Scientific illustration','Infographic','Photograph','Minimal illustration'].map(value=><option key={value}>{value}</option>)}</select></label><button className="quiet-button is-primary" disabled={!!busy} onClick={generate}><UiIcon name="generate"/>Generate illustration</button><p>Uses the enabled Gemini image service when available, otherwise Classio’s existing Pollinations integration. Your description and slide context go to that service. Review scientific details before applying.</p></div>}
   {section==='background'&&<div className="slide-tool-fields"><p>Theme and typography are shared across your presentation. Change them from the presentation’s Theme cards.</p>{draft.image&&<PlacementCards value={draft.imageMode==='background'?'background':draft.imagePlacement} onChange={value=>patch({imageMode:value==='background'?'background':undefined,imagePlacement:value==='background'?'auto':value})}/>}<p>Background images are softened to keep text readable. Source diagrams retain their background unless you explicitly remove it.</p></div>}
  </div><div className="slide-editor-preview"><SlideCanvasEditor slide={draft} theme={theme} layout={layout} design={design} index={index} total={total} onChange={value=>{version.current++;setDraft(value);}} onEditText={editText}/><p>Live preview · double-click text to edit</p></div></div>
  {busy&&<div role="status" className="presentation-progress"><span/>{busy}…<button className="text-button" onClick={()=>{request.current?.abort();setBusy('');}}>Cancel</button></div>}{error&&<p className="tool-error" role="alert">{error}</p>}{notice&&<p role="status">{notice}</p>}
  <footer className="slide-editor-footer"><p>Apply updates this workspace. Account sync status appears in the workspace.</p><button className="quiet-button" onClick={close}>Cancel</button><button className="quiet-button is-primary" disabled={saving||!!busy} onClick={save}>{saving?'Applying…':'Apply changes'}</button></footer>
  {cutout&&<PresentationImageEditor source={draft.image} onClose={()=>setCutout(false)} onApply={image=>patch({imageOriginal:draft.imageOriginal||draft.image,image,imageCrop:null})}/>}
 </div></StudioDialog>;
}
