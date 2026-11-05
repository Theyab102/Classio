import { useCallback, useLayoutEffect, useRef, useState } from "react";

const inkTools = [["pan", "Scroll"], ["pen", "Pen"], ["highlight", "Highlighter"], ["erase", "Eraser"]];

export function useInkTool() {
  const [tool, setTool] = useState("pan");
  const [color, setColor] = useState("#b52332");
  const [size, setSize] = useState(3);
  return { tool, setTool, color, setColor, size, setSize };
}

export function InkToolbar({ ink, strokes, onChange }) {
  return <div className="studio-controls" role="toolbar" aria-label="Annotation tools">
    {inkTools.map(([id, label]) => <button type="button" key={id} aria-pressed={ink.tool === id} onClick={() => ink.setTool(id)}>{label}</button>)}
    <label>Color <input type="color" aria-label="Ink color" value={ink.color} onChange={event => ink.setColor(event.target.value)} /></label>
    <label>Size <input type="range" aria-label="Ink size" min="1" max="12" value={ink.size} onChange={event => ink.setSize(Number(event.target.value))} /></label>
    <button type="button" disabled={!strokes.length} onClick={() => onChange(strokes.slice(0, -1))}>Undo ink</button>
    <button type="button" disabled={!strokes.length} onClick={() => onChange([])}>Clear ink</button>
  </div>;
}

function paintStroke(context, stroke, width, height) {
  const [first, ...rest] = stroke.points;
  if (!first) return;
  const scale = width / 700;
  context.save();
  context.globalCompositeOperation = stroke.tool === "erase" ? "destination-out" : "source-over";
  context.globalAlpha = stroke.tool === "highlight" ? .28 : 1;
  context.strokeStyle = stroke.color;
  context.lineWidth = stroke.size * scale * (stroke.tool === "pen" ? 1 : 4);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(first[0] * width, first[1] * height);
  if (!rest.length) context.lineTo(first[0] * width + .01, first[1] * height + .01);
  for (const [x, y] of rest) context.lineTo(x * width, y * height);
  context.stroke();
  context.restore();
}

export function renderInkImage(strokes) {
  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const context = canvas.getContext("2d");
  for (const stroke of strokes) paintStroke(context, stroke, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

export function InkLayer({ ink, strokes, onChange }) {
  const canvas = useRef(null);
  const active = useRef(null);
  const frame = useRef(null);

  const redraw = useCallback(() => {
    const surface = canvas.current;
    if (!surface) return;
    const { width, height } = surface.getBoundingClientRect();
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2, 2400 / Math.max(width, height));
    const w = Math.max(1, Math.round(width * ratio));
    const h = Math.max(1, Math.round(height * ratio));
    if (surface.width !== w) surface.width = w;
    if (surface.height !== h) surface.height = h;
    const context = surface.getContext("2d");
    context.clearRect(0, 0, w, h);
    for (const stroke of strokes) paintStroke(context, stroke, w, h);
    if (active.current) paintStroke(context, active.current.stroke, w, h);
  }, [strokes]);

  useLayoutEffect(() => {
    redraw();
    const observer = new ResizeObserver(redraw);
    if (canvas.current) observer.observe(canvas.current);
    return () => { observer.disconnect(); cancelAnimationFrame(frame.current); frame.current = null; };
  }, [redraw]);

  function point(event) {
    const bounds = canvas.current.getBoundingClientRect();
    const normalize = value => Math.round(Math.min(1, Math.max(0, value)) * 10000) / 10000;
    return [normalize((event.clientX - bounds.left) / bounds.width), normalize((event.clientY - bounds.top) / bounds.height)];
  }

  function start(event) {
    if (ink.tool === "pan" || active.current || event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    active.current = { pointer: event.pointerId, stroke: { tool: ink.tool, size: ink.size, color: ink.color, points: [point(event)] } };
    redraw();
  }

  function move(event) {
    const session = active.current;
    if (!session || session.pointer !== event.pointerId) return;
    const [x, y] = point(event);
    const points = session.stroke.points;
    const [lastX, lastY] = points[points.length - 1];
    if (Math.hypot(x - lastX, y - lastY) < .001) return;
    if (points.length < 1200) points.push([x, y]);
    if (frame.current === null) frame.current = requestAnimationFrame(() => { frame.current = null; redraw(); });
  }

  function end(event) {
    const session = active.current;
    if (!session || session.pointer !== event.pointerId) return;
    active.current = null;
    onChange([...strokes, session.stroke]);
  }

  return <canvas ref={canvas} aria-label="Ink layer" className="studio-ink" data-tool={ink.tool}
    onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end} />;
}
