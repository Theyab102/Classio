import { InkLayer, InkToolbar, useInkTool } from "./DocumentInk";
import { safeImageSource } from "./documentMedia";
import UiIcon from "./UiIcon";
import StudyMarkdown from "./StudyMarkdown";
import { useState } from "react";
import StudioDialog from "./StudioDialog";
import { mediaPlacement } from "./deckGeometry";
import "./documentStudio.css";

export const deckDesigns = [
  { id: "lecture", name: "Lecture", theme: "teal", description: "A bold dark cover, numbered ideas and focused teaching cards." },
  { id: "editorial", name: "Editorial", theme: "minimal", description: "Paper tones, generous serif type and an open magazine layout." },
  { id: "visual", name: "Visual", theme: "ocean", description: "A graphic cover, a subtle grid and colorful panels beside your images." },
];

export function normalizeSlideTable(rows) {
  if (!Array.isArray(rows) || rows.length < 2 || !Array.isArray(rows[0]) || rows[0].length < 2) return null;
  const columns = Math.min(rows[0].length, 5);
  return rows.slice(0, 8).map(row => Array.from({ length: columns }, (_, index) =>
    String(Array.isArray(row) ? row[index] ?? "" : "").trim().slice(0, 120)));
}

export function parseDeckResponse(raw, count) {
  const cleaned = raw.trim().replace(/^```(?:json)?\s*|\s*```$/gi, "");
  const parsed = JSON.parse(cleaned);
  if (!Array.isArray(parsed.slides) || !parsed.slides.length) throw new Error("The AI returned no slides. Try a shorter topic.");
  const text = (value, max) => typeof value === "string" ? value.trim().slice(0, max) : "";
  const points = values => Array.isArray(values) ? values.filter(value => typeof value === "string").slice(0, 6).map(value => text(value, 280)) : [];
  const side = value => ({ heading: text(value?.heading, 75), bullets: points(value?.bullets) });
  const allowed = new Set(["title", "content", "two-col", "quote", "summary", "table"]);
  const slides = parsed.slides.slice(0, count).map(value => {
    const table = normalizeSlideTable(value?.table);
    let type = allowed.has(value?.type) ? value.type : "content";
    if (type === "table" && !table) type = "content";
    return {
      type, title: text(value?.title, 150) || (type === "quote" ? "" : "Untitled slide"),
      subtitle: text(value?.subtitle, 180), bullets: points(value?.bullets),
      quote: text(value?.quote, 500), author: text(value?.author, 100),
      left: side(value?.left), right: side(value?.right),
      imageSearch: text(value?.imageSearch, 100),
      imageRole: ["diagram","graph","screenshot","photo","object"].includes(value?.imageRole) ? value.imageRole : "diagram",
      ...(["left","right","below"].includes(value?.imagePlacement) ? {imagePlacement:value.imagePlacement} : {}),
      ...(table ? { table } : {}),
      note: text(value?.note, 500),
    };
  });
  return { title: text(parsed.title, 150) || slides[0].title || "Presentation", slides };
}

const list = points => <ul>{(points || []).map((point, index) => <li key={index}><StudyMarkdown inline text={point}/></li>)}</ul>;
const noStrokes = [];

export function DeckSlide({ slide, theme, layout = "lecture", index, total, children }) {
  const image = safeImageSource(slide.image) ? slide.image : null;
  const table = normalizeSlideTable(slide.table);
  const background = image && slide.imageMode === "background";
  const placement=mediaPlacement(slide,index),density=[slide.title,slide.subtitle,...(slide.bullets||[])].join('').length>600?'is-dense':'';
  return <div className={`deck-canvas deck-${layout} deck-kind-${slide.type} ${image&&!background?'deck-has-image media-'+placement:''} ${density}`}
    style={{ "--slide-bg": theme.bg, "--slide-accent": theme.accent, "--slide-text": theme.text, "--slide-sub": theme.sub }}>
    {background && <img className="deck-background" src={image} alt="" referrerPolicy="no-referrer" />}
    <div className="deck-decoration" aria-hidden="true"><span /><span /></div>
    <div className="deck-slide-body">
      <span className="deck-kicker">{layout === "editorial" ? "CLASSIO / FIELD NOTES" : layout === "visual" ? "CLASSIO / EXPLORE" : "CLASSIO / STUDY SESSION"}</span>
      <span className="deck-slide-number">{String(index + 1).padStart(2, "0")} / {String(total).padStart(2, "0")}</span>
      {slide.type === "title" && <span className="deck-cover-index" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>}
      {slide.type === "quote" ? <blockquote><p><StudyMarkdown inline text={'“'+slide.quote+'”'}/></p><cite>{slide.author}</cite></blockquote> : <>
        <h3><StudyMarkdown inline text={slide.title}/></h3>
        <div className={`deck-slide-content ${image && !background ? "deck-with-media" : ""}`}>
          <div>
            {slide.type === "title" ? <p><StudyMarkdown inline text={slide.subtitle}/></p> : slide.type === "two-col" ? <div className="deck-columns">
              <section><h4><StudyMarkdown inline text={slide.left?.heading}/></h4>{list(slide.left?.bullets)}</section>
              <section><h4><StudyMarkdown inline text={slide.right?.heading}/></h4>{list(slide.right?.bullets)}</section>
            </div> : list(slide.bullets)}
            {table && <table className="native-slide-table"><tbody>{table.map((row,i)=><tr key={i}>{row.map((text,j)=>i===0?<th key={j}><StudyMarkdown inline text={text}/></th>:<td key={j}><StudyMarkdown inline text={text}/></td>)}</tr>)}</tbody></table>}
          </div>
          {image && !background && <figure>
            <img src={image} alt={slide.imageAlt || slide.imageSearch || ""} referrerPolicy="no-referrer" />
            {slide.imageCredit && <figcaption>{/^https:\/\//.test(slide.imageSource||"")?<a href={slide.imageSource} target="_blank" rel="noreferrer">{slide.imageCredit}</a>:slide.imageCredit}</figcaption>}
          </figure>}
        </div>
      </>}
    </div>
    {children}
  </div>;
}

export function DeckDesignPicker({ layouts, deck, themes, onChoose, onCancel }) {
  const [preview,setPreview]=useState(null);
  return <section className="deck-choice" aria-label="Presentation design options">
    <h3>Choose a presentation design</h3>
    <p>Compare the same deck in different layouts. Your current slides stay in place until you choose a design.</p>
    <div className="deck-choice-grid">
      {layouts.map(layout => <article key={layout.id}>
        <h4>{layout.name}</h4><p>{layout.description}</p>
        <DeckSlide slide={deck.slides[0]} layout={layout.id} theme={themes.find(theme => theme.id === layout.theme)} index={0} total={deck.slides.length} />
        {deck.slides[1] && <DeckSlide slide={deck.slides[1]} layout={layout.id} theme={themes.find(theme => theme.id === layout.theme)} index={1} total={deck.slides.length} />}
        <div className="studio-controls"><button type="button" className="quiet-button" onClick={()=>setPreview(layout)}>Preview large</button><button type="button" className="quiet-button is-primary" onClick={() => onChoose(layout)}><UiIcon name="check" size={16} />Use {layout.name}</button></div>
      </article>)}
    </div>
    <div className="studio-controls"><button type="button" onClick={onCancel}>Discard options</button></div>
    {preview&&<StudioDialog title={preview.name+" presentation preview"} wide onClose={()=>setPreview(null)}><div className="deck-large-preview">{deck.slides.map((slide,index)=><DeckSlide key={index} slide={slide} layout={preview.id} theme={themes.find(theme=>theme.id===preview.theme)} index={index} total={deck.slides.length}/>)}</div><button className="quiet-button is-primary" onClick={()=>{onChoose(preview);setPreview(null);}}>Use {preview.name}</button></StudioDialog>}
  </section>;
}

export function AnnotatedDeckSlide({ slide, theme, layout, index, total, onChange }) {
  const ink = useInkTool();
  const strokes = slide.ink || noStrokes;
  return <section className="deck-annotation" aria-label="Annotate slide">
    <div className="document-toolbar"><InkToolbar ink={ink} strokes={strokes} onChange={onChange} /></div>
    <DeckSlide slide={slide} theme={theme} layout={layout} index={index} total={total}>
      <InkLayer ink={ink} strokes={strokes} onChange={onChange} />
    </DeckSlide>
  </section>;
}
