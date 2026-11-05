import { InkLayer, InkToolbar, useInkTool } from "./DocumentInk";
import { safeImageSource } from "./documentMedia";
import { SlideTable } from "./PowerPointViewer";
import "./documentStudio.css";

export const deckDesigns = [
  { id: "lecture", name: "Lecture", theme: "teal", description: "Dark cover, prominent titles and high-contrast points." },
  { id: "editorial", name: "Editorial", theme: "minimal", description: "Light paper, serif headings and spacious typography." },
  { id: "visual", name: "Visual", theme: "ocean", description: "An accent rail and side-by-side visuals." },
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
      ...(table ? { table } : {}),
      note: text(value?.note, 500),
    };
  });
  return { title: text(parsed.title, 150) || slides[0].title || "Presentation", slides };
}

const list = points => <ul>{(points || []).map((point, index) => <li key={index}>{point}</li>)}</ul>;
const noStrokes = [];

export function DeckSlide({ slide, theme, layout = "lecture", index, total, children }) {
  const image = safeImageSource(slide.image) ? slide.image : null;
  const table = normalizeSlideTable(slide.table);
  const background = image && slide.imageMode === "background";
  return <div className={`deck-canvas deck-${layout} deck-kind-${slide.type}`}
    style={{ "--slide-bg": theme.bg, "--slide-accent": theme.accent, "--slide-text": theme.text, "--slide-sub": theme.sub }}>
    {background && <img className="deck-background" src={image} alt="" referrerPolicy="no-referrer" />}
    <div className="deck-slide-body">
      <span className="deck-slide-number">{index + 1} / {total}</span>
      {slide.type === "quote" ? <blockquote><p>“{slide.quote}”</p><cite>{slide.author}</cite></blockquote> : <>
        <h3>{slide.title}</h3>
        <div className={`deck-slide-content ${image && !background ? "deck-with-media" : ""}`}>
          <div>
            {slide.type === "title" ? <p>{slide.subtitle}</p> : slide.type === "two-col" ? <div className="deck-columns">
              <section><h4>{slide.left?.heading}</h4>{list(slide.left?.bullets)}</section>
              <section><h4>{slide.right?.heading}</h4>{list(slide.right?.bullets)}</section>
            </div> : list(slide.bullets)}
            {table && <SlideTable table={{ header: true, rows: table.map(row => row.map(text => ({ text }))) }} />}
          </div>
          {image && !background && <figure>
            <img src={image} alt={slide.imageAlt || slide.imageSearch || ""} referrerPolicy="no-referrer" />
            {slide.imageCredit && <figcaption>{slide.imageCredit}</figcaption>}
          </figure>}
        </div>
      </>}
    </div>
    {children}
  </div>;
}

export function DeckDesignPicker({ layouts, deck, themes, onChoose, onCancel }) {
  return <section className="deck-choice" aria-label="Presentation design options">
    <h3>Choose a presentation design</h3>
    <p>Compare the same deck in different layouts. Your current slides stay in place until you choose a design.</p>
    <div className="deck-choice-grid">
      {layouts.map(layout => <article key={layout.id}>
        <h4>{layout.name}</h4><p>{layout.description}</p>
        <DeckSlide slide={deck.slides[0]} layout={layout.id} theme={themes.find(theme => theme.id === layout.theme)} index={0} total={deck.slides.length} />
        {deck.slides[1] && <DeckSlide slide={deck.slides[1]} layout={layout.id} theme={themes.find(theme => theme.id === layout.theme)} index={1} total={deck.slides.length} />}
        <div className="studio-controls"><button type="button" onClick={() => onChoose(layout)}>Use {layout.name}</button></div>
      </article>)}
    </div>
    <div className="studio-controls"><button type="button" onClick={onCancel}>Discard options</button></div>
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
