// Native PowerPoint shapes keep each design editable after export.
export function decorateDeckExport(pptx, slide, { layout, index, total, cover, bg, accent, sub }) {
  const shape = (kind, geometry, color, transparency = 0) => slide.addShape(pptx.ShapeType[kind], { ...geometry, line: { transparency: 100 }, fill: { color, transparency } });
  const fontFace = layout === "editorial" ? "Georgia" : "Arial";
  const x = layout === "editorial" ? 1.05 : .8;
  if (layout === "lecture") {
    shape("rect", { x: 0, y: 7.42, w: 13.333, h: .08 }, accent);
    shape("ellipse", { x: 9.5, y: -2.1, w: 5.3, h: 5.3 }, accent, 92);
  } else if (layout === "editorial") shape("rect", { x, y: .77, w: 11.2, h: .015 }, accent, 35);
  else {
    if (cover) shape("parallelogram", { x: 9.2, y: 0, w: 4.4, h: 7.5 }, accent, 84);
    for (let i = .8; i < 13.3; i += .55) shape("rect", { x: i, y: .9, w: .007, h: 6.2 }, accent, 94);
    for (let i = .9; i < 7.3; i += .55) shape("rect", { x: .8, y: i, w: 12, h: .007 }, accent, 94);
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
