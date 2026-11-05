import { useEffect, useRef, useState } from "react";
import { openPdf } from "./pdf";
import { InkLayer, InkToolbar, useInkTool } from "./DocumentInk";
import "./documentStudio.css";

const noStrokes = [];

export function pdfZoomScale(zoom, width, referenceWidth) {
  return zoom === "fit" ? Math.max(.05, width / referenceWidth) : Number(zoom) / 100 * 96 / 72;
}

export default function LocalPdfViewer({ file, fileObj, onUpdate, explain }) {
  const [document, setDocument] = useState(null);
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(file.pdfZoom || "fit");
  const [width, setWidth] = useState(0);
  const [rendered, setRendered] = useState(null);
  const [error, setError] = useState("");
  const scroll = useRef(null);
  const canvas = useRef(null);
  const ink = useInkTool();
  const strokes = file.pdfInk?.[page] || noStrokes;

  useEffect(() => {
    let live = true;
    let pdf;
    setDocument(null);
    setPage(1);
    setRendered(null);
    setError("");
    (async () => {
      try {
        pdf = await openPdf({ data: new Uint8Array(await fileObj.arrayBuffer()) });
        if (!live) { await pdf.destroy(); return; }
        const first = await pdf.getPage(1);
        if (live) setDocument({ pdf, referenceWidth: first.getViewport({ scale: 1 }).width });
      } catch (failure) {
        if (live) setError(failure.name === "PasswordException"
          ? "This PDF is password-protected. Open an unlocked copy."
          : "Could not open this PDF. Try the original file download.");
      }
    })();
    return () => { live = false; if (pdf) pdf.destroy().catch(() => {}); };
  }, [fileObj]);

  useEffect(() => {
    const view = scroll.current;
    if (!view) return;
    const resize = () => setWidth(Math.max(1, view.clientWidth - 32));
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(view);
    return () => observer.disconnect();
  }, []);

  const scale = document ? pdfZoomScale(zoom, width, document.referenceWidth) : 0;
  useEffect(() => {
    if (!document || !width || !canvas.current) return;
    let live = true;
    let task;
    const target = canvas.current;
    (async () => {
      try {
        const source = await document.pdf.getPage(page);
        if (!live) return;
        const viewport = source.getViewport({ scale });
        const ratio = Math.min(window.devicePixelRatio || 1, 2, 3000 / Math.max(viewport.width, viewport.height));
        const backing = target.ownerDocument.createElement("canvas");
        backing.width = Math.max(1, Math.round(viewport.width * ratio));
        backing.height = Math.max(1, Math.round(viewport.height * ratio));
        task = source.render({
          canvasContext: backing.getContext("2d"), viewport,
          transform: [ratio, 0, 0, ratio, 0, 0],
        });
        await task.promise;
        if (!live) return;
        target.width = backing.width;
        target.height = backing.height;
        target.getContext("2d").drawImage(backing, 0, 0);
        setRendered({ page, scale, width: viewport.width, height: viewport.height });
        setError("");
      } catch (failure) {
        if (live && failure.name !== "RenderingCancelledException") setError("Could not draw this PDF page.");
      }
    })();
    return () => { live = false; task?.cancel(); };
  }, [document, page, scale, width]);

  function goTo(next) {
    const value = Math.max(1, Math.min(document?.pdf.numPages || 1, Math.trunc(next)));
    setPage(value);
    if (scroll.current) scroll.current.scrollTop = 0;
  }

  function changeZoom(value) {
    setZoom(value);
    onUpdate({ ...file, pdfZoom: value });
  }

  async function explainPage() {
    const source = await document.pdf.getPage(page);
    const content = await source.getTextContent();
    return content.items.map(item => item.str || "").join(" ");
  }

  function updateInk(value) {
    onUpdate({ ...file, pdfInk: { ...file.pdfInk, [page]: value } });
  }

  const ready = rendered?.page === page && rendered?.scale === scale;
  const size = rendered ? { width: rendered.width, height: rendered.height } : { width: 1, height: 1 };

  return <section className="document-viewer" aria-label="Original PDF viewer">
    <div className="studio-controls document-toolbar">
      <button type="button" aria-label="Previous PDF page" disabled={page <= 1} onClick={() => goTo(page - 1)}>Previous</button>
      <label>Page <input key={page} type="number" aria-label="PDF page number" min="1" max={document?.pdf.numPages || 1} defaultValue={page}
        onBlur={event => goTo(Number(event.target.value) || page)}
        onKeyDown={event => { if (event.key === "Enter") event.target.blur(); }} /></label>
      <span>of {document?.pdf.numPages || "…"}</span>
      <button type="button" aria-label="Next PDF page" disabled={page >= (document?.pdf.numPages || 1)} onClick={() => goTo(page + 1)}>Next</button>
      <label>Zoom <select aria-label="PDF zoom" value={zoom} onChange={event => changeZoom(event.target.value)}>
        <option value="fit">Fit width</option>{[50, 75, 100, 125, 150, 175, 200].map(value => <option key={value} value={value}>{value}%</option>)}
      </select></label>
      <button type="button" disabled={!ready} onClick={() => explain(explainPage, `PDF page ${page}`)}>Explain page</button>
    </div>
    <div className="document-toolbar"><InkToolbar ink={ink} strokes={strokes} onChange={updateInk} /></div>
    {error && <p className="studio-error" role="alert">{error}</p>}
    {!error && !ready && <p className="pdf-render-status" role="status">{document ? `Rendering page ${page}…` : "Opening the PDF on your device…"}</p>}
    <div ref={scroll} className="pdf-scroll">
      <div className="pdf-page-frame" style={{ ...size, visibility: ready ? "visible" : "hidden" }}>
        <canvas ref={canvas} className="pdf-original-canvas" aria-label={`Original PDF page ${page}`} />
        {ready && <InkLayer key={page} ink={ink} strokes={strokes} onChange={updateInk} />}
      </div>
    </div>
  </section>;
}
