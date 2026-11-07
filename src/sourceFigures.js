import {getPresentationFigures,getPdfFigures,isPdfFile,prepareSourceFigure,likelyStudyImage} from './documentMedia';
import {openPdf} from './pdf';
const catalogCache=new WeakMap(),previewCache=new WeakMap();
export async function getSourceFigures(file){
 if(!file)return[];if(catalogCache.has(file))return catalogCache.get(file);
 const work=(async()=>{
  if(/\.pptx$/i.test(file.name||''))return(await getPresentationFigures(file)).slice(0,40);
  if(isPdfFile(file)){const pdf=await openPdf({data:new Uint8Array(await file.arrayBuffer())});try{const rows=[];for(let i=1;i<=Math.min(pdf.numPages,40);i++){const page=await pdf.getPage(i),content=await page.getTextContent(),text=content.items.map(item=>item.str||'').join(' ');rows.push({pageNumber:i,title:text.slice(0,100)||`PDF page ${i}`,context:text.slice(0,900),credit:`Source PDF · page ${i}`});page.cleanup();}return rows;}finally{await pdf.destroy();}}
  return[];
 })().catch(error=>{catalogCache.delete(file);throw error;});catalogCache.set(file,work);return work;
}
export async function resolveSourceFigure(figure,file){
 if(figure?.src)return prepareSourceFigure(figure);
 if(figure?.pageNumber&&isPdfFile(file)){const image=(await getPdfFigures(file,figure.pageNumber,{pagePreview:true}))[0];return image?{...figure,...image,context:figure.context,credit:figure.credit}:null;}
 return null;
}
export function sourceFigureGuide(figures){return figures.map((image,id)=>`${id}: ${image.title}. ${(image.context||'').slice(0,180)}`).join('\n').slice(0,7500);}
export function chooseSourceFigure(slide,figures,used=new Set()){
 if(Number.isInteger(slide.sourceFigure)&&figures[slide.sourceFigure])return figures[slide.sourceFigure];
 const available=figures.filter(image=>!used.has(image.src||image.pageNumber)),queries=[slide.imageSearch,slide.title,slide.subtitle,...(slide.bullets||[])].filter(Boolean);
 for(const query of queries){const image=likelyStudyImage(query,available);if(image)return image;}
 for(const query of queries){const image=likelyStudyImage(query,figures);if(image)return image;}return null;
}
export async function getSourceThumbnail(file){
 if(!file)return null;if(previewCache.has(file))return previewCache.get(file);
 const work=(async()=>{
  if(isPdfFile(file))return(await getPdfFigures(file,1,{pagePreview:true}))[0];
  if(/\.pptx$/i.test(file.name||'')){const figures=await getSourceFigures(file);if(figures.length)return resolveSourceFigure(figures[0],file);const{readSlides}=await import('./PowerPointViewer');const parsed=await readSlides(file);return{slide:parsed.slides[0],width:parsed.width,height:parsed.height};}
  if(file.type?.startsWith('image/')){const{prepareNoteImage}=await import('./documentMedia');return{src:await prepareNoteImage(file),title:file.name};}
  return null;
 })().catch(()=>{previewCache.delete(file);return null;});previewCache.set(file,work);return work;
}
