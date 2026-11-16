export function deckTypography(slide,geometry){
 const title=slide.type==='title'?54:32;
 const titleLength=String(slide.title||'').length;
 const titleSize=Math.max(22,Math.min(title,titleLength>70?title*.75:title));
 const count=(slide.bullets||[]).join('').length;
 const bodySize=count>700?16:count>450?18:slide.image?20:22;
 return {title:titleSize,body:bodySize,subtitle:20,fontFace:slide.font==='serif'?'Georgia':'Arial'};
}
