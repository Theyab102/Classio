import {normalizeDeckDesign} from "./deckStyle";
// Native PowerPoint shapes keep each design editable after export.
export function decorateDeckExport(pptx, slide, { layout, index, total, cover, bg, accent, sub,design }) {
  const shape = (kind, geometry, color, transparency = 0) => slide.addShape(pptx.ShapeType[kind], { ...geometry, line: { transparency: 100 }, fill: { color, transparency } });
  const style=design?normalizeDeckDesign(design):null;
  const fontFace=style?.font==="serif"?"Georgia":style?.font==="handwritten"?"Comic Sans MS":style?.font==="sans"?"Arial":layout==="editorial"?"Georgia":"Arial";
  if(style&&style.pattern!=="none"){
    const color=style.patternColor.slice(1),opacity=100-style.patternOpacity*100,step=.52;
    switch(style.pattern){
      case "checkerboard":
        for(let row=0;row<15;row++)for(let col=0;col<26;col++)if((row+col)%2===0)shape("rect",{x:col*step,y:row*step,w:Math.min(step,13.333-col*step),h:Math.min(step,7.5-row*step)},color,opacity);
        break;
      case "dots":
        for(let y=.3;y<7.5;y+=.45)for(let x=.3;x<13.3;x+=.45)shape("ellipse",{x,y,w:.04,h:.04},color,opacity);
        break;
      case "stripes":
        for(let x=-7;x<14;x+=.6)slide.addShape(pptx.ShapeType.line,{x,y:0,w:7.5,h:7.5,line:{color,transparency:opacity,width:9}});
        break;
      default:
        for(let y=.9;y<7.3;y+=.45)shape("rect",{x:0,y,w:13.333,h:.009},color,opacity);
        if(style.pattern==="grid")for(let x=.4;x<13.3;x+=.45)shape("rect",{x,y:0,w:.009,h:7.5},color,opacity);
    }
  }
  const x = layout === "editorial" ? 1.05 : .8;
  if (!style||style.pattern==="none") {
  if (layout === "lecture") {
    shape("rect", { x: 0, y: 7.42, w: 13.333, h: .08 }, accent);
    shape("ellipse", { x: 9.5, y: -2.1, w: 5.3, h: 5.3 }, accent, 92);
  } else if (layout === "editorial") shape("rect", { x, y: .77, w: 11.2, h: .015 }, accent, 35);
  else {
    if (cover) shape("parallelogram", { x: 9.2, y: 0, w: 4.4, h: 7.5 }, accent, 84);
    for (let i = .8; i < 13.3; i += .55) shape("rect", { x: i, y: .9, w: .007, h: 6.2 }, accent, 94);
    for (let i = .9; i < 7.3; i += .55) shape("rect", { x: .8, y: i, w: 12, h: .007 }, accent, 94);
  }
  }
  slide.addText(layout === "editorial" ? "CLASSIO / FIELD NOTES" : layout === "visual" ? "CLASSIO / EXPLORE" : "CLASSIO / STUDY SESSION", { x, y: .32, w: 6.5, h: .25, fontSize: 10, charSpacing: 2, bold: true, color: accent, fontFace: "Arial", margin: 0 });
  slide.addText(String(index + 1).padStart(2, "0") + " / " + String(total).padStart(2, "0"), { x: 11.2, y: .32, w: 1.1, h: .25, fontSize: 10, color: sub, align: "right", margin: 0 });
  if (cover) slide.addText(String(index + 1).padStart(2, "0"), { x: 9.3, y: 3.6, w: 3.2, h: 2.8, fontSize: layout === "visual" ? 200 : 170, fontFace, color: accent, transparency: layout === "visual" ? 70 : 90, margin: 0, bold: layout !== "editorial" });
  return { x, fontFace, bg };
}

export function addDeckExportPoints(pptx, slide, points, { layout, x, width, accent, bg, text, sub, fontFace, hasImage, y:startY=1.94, height:availableHeight=4.45 }) {
  const columns = hasImage ? 1 : 2, gap = .22, w = (width - gap * (columns - 1)) / columns;
  const height = Math.min(1.2, availableHeight / Math.max(1, Math.ceil(points.length / columns)) - .16);
  points.forEach((point, index) => {
    const cx = x + index % columns * (w + gap), y = startY + Math.floor(index / columns) * (height + .16);
    if (layout === "editorial") {
      slide.addShape(pptx.ShapeType.rect, { x: cx, y, w, h: .014, line: { transparency: 100 }, fill: { color: accent, transparency: 55 } });
      slide.addText(point, { x: cx, y: y + .15, w, h: height - .15, fontSize: 20, fontFace, color: sub, fit: "shrink", valign: "top", margin: 0 });
    } else {
      const strong = layout === "visual" && index % 2 === 0;
      slide.addShape(pptx.ShapeType.roundRect, { x: cx, y, w, h: height, radius: .12,
        line: { color: accent, transparency: strong ? 100 : 65, pt: .65 }, fill: { color: accent, transparency: strong ? 0 : 92 } });
      if (layout === "lecture") slide.addText(String(index + 1).padStart(2, "0"), { x: cx + .14, y: y + .21, w: .34, h: .3, fontSize: 12, bold: true, color: accent, fontFace, margin: 0 });
      slide.addText(point, { x: cx + (layout === "lecture" ? .55 : .21), y: y + .16, w: w - (layout === "lecture" ? .75 : .42), h: height - .3, fontSize: 19, color: strong ? bg : text, fontFace, fit: "shrink", valign: "mid", margin: 0 });
    }
  });
}
