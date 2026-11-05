import { useEffect, useState } from "react";
import { InkLayer, InkToolbar, useInkTool } from "./DocumentInk";
import "./documentStudio.css";

const noStrokes = [];
const first = (node, name) => node?.getElementsByTagNameNS("*", name)[0];
const all = (node, name) => Array.from(node?.getElementsByTagNameNS("*", name) || []);
const number = (node, name) => Number(node?.getAttribute(name));
const parse = text => new DOMParser().parseFromString(text, "application/xml");
const plain = node => all(node, "t").map(element => element.textContent).join("");

export function parsePowerPointTables(xml) {
  return all(xml, "tbl").map(table => {
    const columns = all(first(table, "tblGrid"), "gridCol").map(col => number(col, "w"));
    const header = ["1", "true"].includes(first(table, "tblPr")?.getAttribute("firstRow"));
    return {
      columns,
      header,
      rows: all(table, "tr").map(row => all(row, "tc")
        .filter(cell => !["1", "true"].includes(cell.getAttribute("hMerge")) && !["1", "true"].includes(cell.getAttribute("vMerge")))
        .map(cell => ({
          text: all(cell, "p").map(plain).join("\n"),
          colSpan: Math.max(1, number(cell, "gridSpan")),
          rowSpan: Math.max(1, number(cell, "rowSpan")),
          fill: color(first(first(cell, "tcPr"), "solidFill")),
        }))),
    };
  });
}

function color(node) {
  const value = first(node, "srgbClr")?.getAttribute("val");
  return /^[0-9a-f]{6}$/i.test(value || "") ? "#" + value : undefined;
}

function geometry(shape, width, height) {
  const transform = first(shape, "xfrm");
  const offset = first(transform, "off");
  const extent = first(transform, "ext");
  if (!extent || !offset || !number(extent, "cx") || !number(extent, "cy")) return null;
  return {
    left: `${number(offset, "x") / width * 100}%`, top: `${number(offset, "y") / height * 100}%`,
    width: `${number(extent, "cx") / width * 100}%`, height: `${number(extent, "cy") / height * 100}%`,
    transform: `rotate(${number(transform, "rot") / 60000}deg)`,
  };
}

const relationId = element => Array.from(element?.attributes || []).find(attribute => attribute.localName === "id" && attribute.namespaceURI?.endsWith("/relationships"))?.value;
const relsFor = (zip, name) => {
  const path = name.replace(/([^/]+)$/, "_rels/$1.rels");
  return zip.file(path)?.async("string").then(xml => Object.fromEntries(all(parse(xml), "Relationship").map(item => [item.getAttribute("Id"), item.getAttribute("Target")]))) || {};
};
const resolved = (path, target) => new URL(target, `https://classio.local/${path}`).pathname.slice(1);

async function readSlides(file) {
  const { default: JSZip } = await import("jszip");
  const zip = await JSZip.loadAsync(await file.arrayBuffer());
  const presentationName = "ppt/presentation.xml";
  const presentationFile = zip.file(presentationName);
  if (!presentationFile) throw new Error("Use a .pptx presentation.");
  const presentation = parse(await presentationFile.async("string"));
  const presentationRels = await relsFor(zip, presentationName);
  const dimensions = first(presentation, "sldSz");
  const width = number(dimensions, "cx") || 12192000;
  const height = number(dimensions, "cy") || 6858000;
  const paths = all(presentation, "sldId").map(item => presentationRels[relationId(item)])
    .filter(Boolean).map(target => resolved(presentationName, target));
  if (!paths.length) throw new Error("No slides found.");

  const slides = await Promise.all(paths.map(async path => {
    const xml = parse(await zip.file(path).async("string"));
    const rels = await relsFor(zip, path);
    const layoutTarget = Object.values(rels).find(target => /slideLayouts\//.test(target));
    const layoutFile = layoutTarget && zip.file(resolved(path, layoutTarget));
    const layout = layoutFile ? parse(await layoutFile.async("string")) : null;
    const placeholders = all(layout, "sp").filter(shape => first(shape, "ph"));
    const tree = first(xml, "spTree");
    const items = [];
    for (const shape of tree?.children || []) {
      const name = shape.localName;
      const placeholder = first(shape, "ph");
      const match = placeholder && placeholders.find(item => {
        const other = first(item, "ph");
        return placeholder.hasAttribute("idx")
          ? placeholder.getAttribute("idx") === other.getAttribute("idx")
          : placeholder.getAttribute("type") === other.getAttribute("type");
      });
      const position = geometry(shape, width, height) || (match && geometry(match, width, height));
      if (name === "sp") {
        const paragraphs = all(first(shape, "txBody"), "p").map(paragraph => {
          const props = first(paragraph, "rPr") || first(paragraph, "defRPr");
          const points = number(props, "sz") / 100;
          return {
            text: plain(paragraph),
            size: points > 0 ? `${points * 12700 / width * 100}cqw` : undefined,
            bold: props?.getAttribute("b") === "1",
            color: color(first(props, "solidFill")),
          };
        }).filter(paragraph => paragraph.text);
        if (paragraphs.length) items.push({ type: "text", position, paragraphs, title: ["title", "ctrTitle"].includes(placeholder?.getAttribute("type")) });
      } else if (name === "pic") {
        const blip = first(shape, "blip");
        const id = Array.from(blip?.attributes || []).find(attribute => attribute.localName === "embed")?.value;
        const image = rels[id] && resolved(path, rels[id]);
        const ext = image?.split(".").pop().toLowerCase();
        if (image && ["png", "jpg", "jpeg", "gif", "webp"].includes(ext) && zip.file(image)) {
          const base64 = await zip.file(image).async("base64");
          items.push({ type: "image", position, src: `data:image/${ext === "jpg" ? "jpeg" : ext};base64,${base64}` });
        }
      } else if (name === "graphicFrame") {
        for (const table of parsePowerPointTables(shape)) items.push({ type: "table", position, table });
      }
    }
    return {
      items,
      background: color(first(first(xml, "bgPr"), "solidFill")) || "#ffffff",
      texts: items.flatMap(item => item.type === "text" ? item.paragraphs.map(paragraph => paragraph.text)
        : item.type === "table" ? item.table.rows.flatMap(row => row.map(cell => cell.text)) : []),
    };
  }));
  return { slides, width, height };
}

export function SlideTable({ table }) {
  const columnWidth = table.columns?.reduce((sum, value) => sum + value, 0);
  return <table className="native-slide-table">
    {columnWidth > 0 && <colgroup>{table.columns.map((value, index) => <col key={index} style={{ width: `${value / columnWidth * 100}%` }} />)}</colgroup>}
    <tbody>{table.rows.map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, index) => {
      const Tag = rowIndex === 0 && table.header ? "th" : "td";
      return <Tag key={index} colSpan={cell.colSpan} rowSpan={cell.rowSpan} style={{ backgroundColor: cell.fill }}>{cell.text}</Tag>;
    })}</tr>)}</tbody>
  </table>;
}

function SlideItem({ item }) {
  return <div className={item.position ? "ppt-positioned" : "ppt-flow-item"} style={item.position || undefined}>
    {item.type === "text" && item.paragraphs.map((paragraph, index) => <p key={index}
      style={{ fontSize: paragraph.size || (item.title ? "4cqw" : "2.3cqw"), fontWeight: paragraph.bold || item.title ? 700 : undefined, color: paragraph.color }}>
      {paragraph.text}
    </p>)}
    {item.type === "image" && <img src={item.src} alt="" />}
    {item.type === "table" && <SlideTable table={item.table} />}
  </div>;
}

export default function PowerPointViewer({ file = {}, fileObj, page, onTotalPages, onSlidesLoaded, onUpdate }) {
  const [deck, setDeck] = useState(null);
  const [error, setError] = useState("");
  const ink = useInkTool();
  const strokes = file.powerPointInk?.[page] || noStrokes;
  const { slides, width, height } = deck || {};
  const slide = slides?.[page - 1];
  const positioned = slide?.items.every(item => item.position);

  useEffect(() => {
    let live = true;
    setError("");
    setDeck(null);
    readSlides(fileObj).then(result => {
      if (!live) return;
      setDeck(result);
      onTotalPages?.(result.slides.length);
      onSlidesLoaded?.(result.slides);
    }).catch(failure => { if (live) setError(failure.message || "Could not read this presentation."); });
    return () => { live = false; };
  // Callbacks only publish the parsed file; changing them does not require parsing it again.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileObj]);

  const updateInk = value => onUpdate({ ...file, powerPointInk: { ...file.powerPointInk, [page]: value } });

  if (error) return <p role="alert" className="studio-error">{error}</p>;
  if (!slide) return <p role="status" className="studio-status">Opening the PowerPoint file on your device…</p>;

  return <section className="document-viewer ppt-document" aria-label="PowerPoint slide viewer">
    {onUpdate && <div className="document-toolbar"><InkToolbar ink={ink} strokes={strokes} onChange={updateInk} /></div>}
    <div className={`ppt-page ${positioned ? "ppt-positioned-page" : ""}`}
      style={{ aspectRatio: `${width} / ${height}`, background: slide.background }} aria-label={`PowerPoint slide ${page}`}>
      {slide.items.map((item, index) => <SlideItem key={index} item={positioned ? item : { ...item, position: null }} />)}
      {!slide.items.length && <p>No supported slide content was found.</p>}
      {onUpdate && <InkLayer key={page} ink={ink} strokes={strokes} onChange={updateInk} />}
    </div>
  </section>;
}
