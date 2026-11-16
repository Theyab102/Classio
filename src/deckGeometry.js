export function mediaPlacement(slide,index=0){
  if(['left','right','below','center','full'].includes(slide.imagePlacement))return slide.imagePlacement;
  if(slide.imageAspect>1.85&&(slide.bullets?.length||0)<=3&&slide.type!=='table')return 'below';
  return slide.type!=='title'&&index%3===2?'left':'right';
}
export function deckGeometry(slide,layout='lecture',index=0){
  const margin=layout==='editorial'?.95:layout==='visual'?1.05:.6,width=13.333-margin*2,cover=slide.type==='title',hasImage=!!slide.image&&slide.imageMode!=='background',placement=mediaPlacement(slide,index);
  let title={x:margin,y:cover?2.0:1.0,w:width,h:cover?1.7:.9},body={x:margin,y:cover?4.0:2.15,w:width,h:cover?1.1:4.55},image=null;
  if(hasImage&&placement==='below'){title={x:margin,y:cover?1.4:1,w:width,h:cover?1.2:.85};body={x:margin,y:cover?2.65:2.05,w:width,h:cover?.7:2};image={x:margin,y:4.25,w:width,h:2.35};}
  else if(hasImage){const figureWidth=width*.36,textWidth=width*.58,gap=width*.06,textX=placement==='left'?margin+figureWidth+gap:margin;title.w=textWidth;title.x=textX;body.w=textWidth;body.x=textX;image={x:placement==='left'?margin:margin+textWidth+gap,y:cover?1.8:2.15,w:figureWidth,h:cover?4.7:4.4};}
  if(hasImage&&['center','full'].includes(placement)){
    title={x:margin,y:.9,w:width,h:.85};body={x:margin,y:6.45,w:width,h:.6};
    image={x:placement==='full'?.25:margin,y:1.95,w:placement==='full'?12.833:width,h:4.25};
  }
  if(slide.type==='quote')body={x:margin,y:2.2,w:width,h:2.7};
  const bounded=box=>{const w=Math.max(.35,Math.min(13.333,Number(box.w)||.35)),h=Math.max(.25,Math.min(7.5,Number(box.h)||.25));return {x:Math.max(0,Math.min(13.333-w,Number(box.x)||0)),y:Math.max(0,Math.min(7.5-h,Number(box.y)||0)),w,h};};
  for(const [key,box] of Object.entries(slide.geometry||{})){
    if(key==='title')title=bounded({...title,...box});
    if(key==='body')body=bounded({...body,...box});
    if(key==='image'&&image)image=bounded({...image,...box});
  }
  return {title,body,image,placement,margin};
}
