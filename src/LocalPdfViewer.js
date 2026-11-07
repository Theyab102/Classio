import { useEffect, useRef, useState } from "react";
import { openPdf } from "./pdf";
import { useStudyActivity } from "./LearningSession";
import UiIcon from "./UiIcon";
import { InkLayer, InkToolbar, useInkTool } from "./DocumentInk";
import "./documentStudio.css";

const noStrokes = [];

export function pdfZoomScale(zoom, width, referenceWidth) {
  return zoom === "fit" ? Math.max(.05, width / referenceWidth) : Number(zoom) / 100 * 96 / 72;
}

export default function LocalPdfViewer({ file, fileObj, onUpdate, explain }) {
  const [document, setDocument] = useState(null);
  const [page, setPage] = useState(file.pdfPage || 1);
  const [zoom, setZoom] = useState(file.pdfZoom || "fit");
  const [width, setWidth] = useState(0);
  const [rendered, setRendered] = useState(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState(""), [matches, setMatches] = useState([]), [searching, setSearching] = useState(false), [pageText, setPageText] = useState("");
  const [textItems, setTextItems] = useState([]), [pageViewport, setPageViewport] = useState(null);
  const textCache = useRef(new Map());
  const savedPage = useRef(file.pdfPage || 1);
  savedPage.current = file.pdfPage || 1;
  useStudyActivity(file.id, "PDF page", pageText, `PDF page ${page}`);
  const scroll = useRef(null);
  const canvas = useRef(null);
  const ink = useInkTool();
  const strokes = file.pdfInk?.[page] || noStrokes;

  useEffect(() => {
    let live = true;
    let pdf;
    setDocument(null);
    setPage(savedPage.current);
    textCache.current.clear();
    setRendered(null);
    setError("");
    (async () => {
      try {
        pdf = await openPdf({ data: new Uint8Array(await fileObj.arrayBuffer()) });
        if (!live) { await pdf.destroy(); return; }
        const first = await pdf.getPage(1);
        if (live) { setPage(value => Math.max(1, Math.min(pdf.numPages, value))); setDocument({ pdf, referenceWidth: first.getViewport({ scale: 1 }).width }); }
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
        const text = textCache.current.get(page) || await source.getTextContent();
        textCache.current.set(page, text);
        if (!live) return;
        setPageText(text.items.map(item => item.str || "").join(" ")); setTextItems(text.items); setPageViewport(viewport);
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

  useEffect(() => {
    let active = true;
    if (!document || query.trim().length < 2) { setMatches([]); setSearching(false); return; }
    setSearching(true);
    const timer = setTimeout(async () => {
      const found = [], term = query.toLocaleLowerCase();
      try {
        for (let i = 1; i <= document.pdf.numPages; i++) {
          if (!active) return;
          const content = textCache.current.get(i) || await (await document.pdf.getPage(i)).getTextContent();
          textCache.current.set(i, content);
          const text = content.items.map(item => item.str || "").join(" "), position = text.toLocaleLowerCase().indexOf(term);
          if (position >= 0) found.push({ page: i, snippet: text.slice(Math.max(0, position - 35), position + 95) });
        }
        if (active) setMatches(found);
      } catch { if (active) setError("The PDF search could not finish. You can still view its pages."); }
      finally { if (active) setSearching(false); }
    }, 300);
    return () => { active = false; clearTimeout(timer); };
  }, [document, query]);

  function goTo(next) {
    const value = Math.max(1, Math.min(document?.pdf.numPages || 1, Math.trunc(next)));
    setPage(value); onUpdate({ ...file, pdfPage: value });
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
      <label className="document-search"><UiIcon name="search" /><input type="search" aria-label="Search PDF" placeholder="Find in PDF" value={query} onChange={event => setQuery(event.target.value)} /></label>
      <button type="button" disabled={!ready} onClick={() => explain(explainPage, `PDF page ${page}`)}>Explain page</button>
    </div>
    <div className="document-toolbar"><InkToolbar ink={ink} strokes={strokes} onChange={updateInk} /></div>
    {query.trim().length >= 2 && <div className="document-search-results" aria-live="polite"><span>{searching ? "Searching the document…" : `${matches.length} matching pages`}</span>{matches.map(match => <button type="button" key={match.page} className="pdf-search-result" title={match.snippet} onClick={() => goTo(match.page)}>Page {match.page}<span>{match.snippet}</span></button>)}</div>}
    {error && <p className="studio-error" role="alert">{error}</p>}
    {!error && !ready && <p className="pdf-render-status" role="status">{document ? `Rendering page ${page}…` : "Opening the PDF on your device…"}</p>}
    <div ref={scroll} className="pdf-scroll">
      <div className="pdf-page-frame" style={{ ...size, visibility: ready ? "visible" : "hidden" }}>
        <canvas ref={canvas} className="pdf-original-canvas" aria-label={`Original PDF page ${page}`} />
        {ready && query.trim().length >= 2 && pageViewport && <div className="pdf-search-highlights" aria-hidden="true">{textItems.filter(item => item.str?.toLocaleLowerCase().includes(query.toLocaleLowerCase()) && item.transform).map((item, index) => {
          const rect = pageViewport.convertToViewportRectangle([item.transform[4], item.transform[5], item.transform[4] + item.width, item.transform[5] + (item.height || 10)]);
          return <span key={index} style={{ left: Math.min(rect[0], rect[2]), top: Math.min(rect[1], rect[3]), width: Math.abs(rect[2] - rect[0]), height: Math.abs(rect[3] - rect[1]) }} />;
        })}</div>}
        {ready && <InkLayer key={page} ink={ink} strokes={strokes} onChange={updateInk} />}
      </div>
    </div>
  </section>;
}
