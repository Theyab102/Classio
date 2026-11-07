export async function imageDataForEditing(source,signal){
  if(/^data:image\/(?:png|jpeg|webp);base64,/i.test(source||''))return source;
  if(!/^https:\/\//i.test(source||''))throw new Error('Choose an uploaded or supported source image.');
  let response;try{response=await fetch(source,{signal,referrerPolicy:'no-referrer'});}catch{throw new Error('This website does not allow image editing from Classio. Upload your copy of the image to edit it.');}
  if(!response.ok)throw new Error('The source image is unavailable. The original is kept.');const blob=await response.blob();if(!/^image\/(?:png|jpeg|webp)$/.test(blob.type)||blob.size>3000000)throw new Error('Upload a PNG, JPEG or WebP image under 3 MB.');
  return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('Could not read the image.'));reader.readAsDataURL(blob);});
}
export async function compactCutout(source){
  const image=new Image();image.src=source;await image.decode();const canvas=document.createElement('canvas'),scale=Math.min(1,800/Math.max(image.width,image.height));canvas.width=Math.max(1,Math.round(image.width*scale));canvas.height=Math.max(1,Math.round(image.height*scale));const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0,canvas.width,canvas.height);
  const pixels=ctx.getImageData(0,0,canvas.width,canvas.height).data;let clear=false,subject=false;for(let i=3;i<pixels.length;i+=4){if(pixels[i]<200)clear=true;if(pixels[i]>200)subject=true;}if(!clear||!subject)throw new Error('The service did not return a usable transparent cutout. The original image is kept.');
  for(const quality of[.9,.75,.6]){const data=canvas.toDataURL('image/webp',quality);if(data.length<200000)return data;}throw new Error('This cutout is too detailed to save safely. Use a smaller image.');
}
export function mayRemoveBackground(slide,figure){return slide.imageRole==='object'&&!/diagram|graph|chart|screenshot|trajectory|formula|equation|illustration|labelled|labeled/i.test([slide.title,slide.imageSearch,figure?.title].join(' '));}
export async function powerpointImage(source){if(!source.startsWith('data:image/webp'))return source;const image=new Image();image.src=source;await image.decode();const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').drawImage(image,0,0);return canvas.toDataURL('image/png');}
