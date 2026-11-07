import {useEffect,useState,lazy,Suspense} from 'react';
const SlidePreview=lazy(()=>import('./PowerPointViewer').then(module=>({default:module.SourceSlidePreview})));
export default function FileThumbnail({file,getFileBlob,children}){
 const[preview,setPreview]=useState(null),blob=file._fileObj||getFileBlob?.(file);
 useEffect(()=>{let live=true;setPreview(null);if(blob)import("./sourceFigures").then(module=>module.getSourceThumbnail(blob)).then(value=>{if(live)setPreview(value);}).catch(()=>{});return()=>{live=false;};},[blob]);
 if(preview?.src)return <div className="material-source-thumbnail" data-source-thumbnail="image"><img src={preview.src} alt={`Preview from ${file.name}`} loading="lazy" draggable="false"/></div>;
 if(preview?.slide)return <div className="material-source-thumbnail" data-source-thumbnail="slide"><Suspense fallback={children}><SlidePreview slide={preview.slide} width={preview.width} height={preview.height}/></Suspense></div>;
 return children;
}
