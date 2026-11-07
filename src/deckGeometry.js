export function mediaPlacement(slide,index=0){
  if(['left','right','below'].includes(slide.imagePlacement))return slide.imagePlacement;
  if(slide.imageAspect>1.85&&(slide.bullets?.length||0)<=3&&slide.type!=='table')return 'below';
  return slide.type!=='title'&&index%3===2?'left':'right';
}
export function deckGeometry(slide,layout='lecture',index=0){
  const margin=layout==='editorial'?.95:layout==='visual'?1.05:.6,width=13.333-margin*2,cover=slide.type==='title',hasImage=!!slide.image&&slide.imageMode!=='background',placement=mediaPlacement(slide,index);
  let title={x:margin,y:cover?2.0:1.0,w:width,h:cover?1.7:.9},body={x:margin,y:cover?4.0:2.15,w:width,h:cover?1.1:4.55},image=null;
  if(hasImage&&placement==='below'){title={x:margin,y:cover?1.4:1,w:width,h:cover?1.2:.85};body={x:margin,y:cover?2.65:2.05,w:width,h:cover?.7:2};image={x:margin,y:4.25,w:width,h:2.35};}
  else if(hasImage){const figureWidth=width*.36,textWidth=width*.58,gap=width*.06,textX=placement==='left'?margin+figureWidth+gap:margin;title.w=textWidth;title.x=textX;body.w=textWidth;body.x=textX;image={x:placement==='left'?margin:margin+textWidth+gap,y:cover?1.8:2.15,w:figureWidth,h:cover?4.7:4.4};}
  return {title,body,image,placement,margin};
}
