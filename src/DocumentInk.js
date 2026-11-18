import ColorField from "./ColorPicker";
import { useCallback, useLayoutEffect, useRef, useState } from "react";
import UiIcon from "./UiIcon";

const inkTools = [["pan", "Scroll"], ["pen", "Pen"], ["highlight", "Highlighter"], ["erase", "Eraser"]];

export function useInkTool(scopeKey) {
  const [tool, setTool] = useState("pan");
  const [color, setColor] = useState("#b52332");
  const [size, setSize] = useState(3);
  const [fingerDrawing,setFingerDrawing]=useState(false);
  const [future,setFuture]=useState([]),expected=useRef(null);
  const [past,setPast]=useState([]),scope=useRef(scopeKey);
  useLayoutEffect(()=>{if(scope.current!==scopeKey){scope.current=scopeKey;setPast([]);setFuture([]);expected.current=null;}},[scopeKey]);
  return { tool, setTool, color, setColor, size, setSize,fingerDrawing,setFingerDrawing,future,setFuture,expected,past,setPast };
}

export function InkToolbar({ ink, strokes, onChange }) {
  const [options,setOptions]=useState(()=>!!strokes.length);
  return <div className="studio-controls" role="toolbar" aria-label="Annotation tools">
    <div className="ink-history" role="group" aria-label="Drawing history">
      <button type="button" aria-label="Undo ink" title="Undo ink (Ctrl/⌘ Z)" disabled={!strokes.length&&!ink.past.length} onClick={()=>undoInk(ink,strokes,onChange)}><UiIcon name="undo" size={21}/><span>Undo</span></button>
      <button type="button" aria-label="Redo ink" title="Redo ink (Ctrl/⌘ Shift Z)" disabled={!ink.future.length} onClick={()=>redoInk(ink,strokes,onChange)}><UiIcon name="redo" size={21}/><span>Redo</span></button>
    </div>
    <div className="ink-tool-choices">{inkTools.map(([id, label]) => <button type="button" key={id} aria-pressed={ink.tool === id} onClick={() => ink.setTool(id)}><UiIcon name={id === "pen" ? "edit" : id === "highlight" ? "highlight" : id === "erase" ? "eraser" : "hand"} size={16} />{label}</button>)}</div>
    <button type="button" aria-expanded={options||ink.tool!=="pan"} className="ink-options-toggle" onClick={()=>{if(ink.tool!=="pan"){ink.setTool("pan");setOptions(false);}else setOptions(open=>!open);}}>Drawing options</button>
    <div className="ink-tool-details studio-controls" hidden={!options&&ink.tool==="pan"}><div className="ink-colors" role="group" aria-label="Ink colors">{[["#b52332", "Red"], ["#286a9b", "Blue"], ["#2f7b59", "Green"], ["#d19a16", "Gold"], ["#7b54a0", "Purple"]].map(([color, name]) => <button type="button" key={color} className="ink-color-swatch" aria-label={name + " ink"} aria-pressed={ink.color === color} style={{ "--swatch": color }} onClick={() => ink.setColor(color)}>{ink.color === color && <UiIcon name="check" size={13} />}</button>)}<label className="ink-custom-color" title="Custom ink color"><span style={{ background: ink.color }} /><ColorField aria-label="Ink color" value={ink.color} onChange={event => ink.setColor(event.target.value)} /></label></div>
    <div className="ink-size-choices" role="group" aria-label="Ink size">{[[2, "Fine"], [5, "Medium"], [10, "Broad"]].map(([size, name]) => <button type="button" key={size} aria-label={name + " stroke"} aria-pressed={size === (ink.size <= 3 ? 2 : ink.size <= 7 ? 5 : 10)} onClick={() => ink.setSize(size)}><span style={{ width: size + 3, height: size + 3 }} /><small>{name}</small></button>)}</div>
    <label className="ink-finger-option"><input type="checkbox" checked={ink.fingerDrawing} onChange={event=>ink.setFingerDrawing(event.target.checked)}/>Draw with finger</label>
    <button type="button" disabled={!strokes.length} onClick={()=>commitInk(ink,strokes,[],onChange)}>Clear ink</button></div>
  </div>;
}

function commitInk(ink,previous,next,onChange){ink.setPast(past=>[...past,previous].slice(-40));ink.expected.current=next;ink.setFuture([]);onChange(next);}
function undoInk(ink,strokes,onChange){if(!strokes.length&&!ink.past.length)return;const next=ink.past.length?ink.past.at(-1):strokes.slice(0,-1);ink.setPast(ink.past.slice(0,-1));ink.setFuture([strokes,...ink.future]);ink.expected.current=next;onChange(next);}
function redoInk(ink,strokes,onChange){if(!ink.future.length)return;const next=ink.future[0];ink.setPast([...ink.past,strokes].slice(-40));ink.setFuture(ink.future.slice(1));ink.expected.current=next;onChange(next);}

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
  const touches=useRef(new Set());
  const latestStrokes=useRef(strokes);
  useLayoutEffect(()=>{if(ink.expected && ink.expected.current!==strokes){ink.setFuture([]);ink.setPast([]);}if(ink.expected)ink.expected.current=strokes;},[strokes,ink]);

  const redraw = useCallback(() => {
    const surface = canvas.current;
    if (!surface) return;
    const context = surface.getContext("2d");
    // Clear the old preview even during a layout transition with zero bounds.
    context.clearRect(0, 0, surface.width, surface.height);
    const { width, height } = surface.getBoundingClientRect();
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2, 2400 / Math.max(width, height));
    const w = Math.max(1, Math.round(width * ratio));
    const h = Math.max(1, Math.round(height * ratio));
    if (surface.width !== w) surface.width = w;
    if (surface.height !== h) surface.height = h;
    for (const stroke of latestStrokes.current) paintStroke(context, stroke, w, h);
    if (active.current) paintStroke(context, active.current.stroke, w, h);
  }, []);

  // ResizeObserver/animation callbacks can already be queued when Undo changes
  // the workspace. Always paint the current history, never a captured old page.
  useLayoutEffect(()=>{latestStrokes.current=strokes;redraw();},[strokes,redraw,ink]);

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
    if(event.pointerType==="touch"){
      touches.current.add(event.pointerId);
      if(touches.current.size>1){active.current=null;redraw();return;}
      if(!ink.fingerDrawing)return;
    }
    if (ink.tool === "pan" || active.current || event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.focus({preventScroll:true});
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
    touches.current.delete(event.pointerId);
    const session = active.current;
    if (!session || session.pointer !== event.pointerId) return;
    active.current = null;
    commitInk(ink,strokes,[...strokes,session.stroke],onChange);
  }

  function cancel(event){touches.current.delete(event.pointerId);if(active.current?.pointer===event.pointerId){active.current=null;redraw();}}

  return <canvas tabIndex={0} onKeyDown={event=>{if(!(event.ctrlKey||event.metaKey)||event.altKey||!["z","y"].includes(event.key.toLowerCase()))return;event.preventDefault();event.stopPropagation();const redo=event.key.toLowerCase()==="y"||event.shiftKey;if(redo)redoInk(ink,strokes,onChange);else undoInk(ink,strokes,onChange);}} ref={canvas} aria-label="Ink layer" className="studio-ink" data-tool={ink.tool}
    data-finger-drawing={ink.fingerDrawing} onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerCancel={cancel} onLostPointerCapture={cancel} />;
}
