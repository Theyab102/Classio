import { MATH_OUTPUT_GUIDE } from "./mathMarkdown";
import { useEffect, useRef, useState } from "react";
import { downloadStudyFile, makeStudyId, scheduleCardReview, sendStudyRequest } from "./studyTools";
import { useStudyActivity } from "./LearningSession";
import StudioDialog from "./StudioDialog";
import StudyMarkdown from "./StudyMarkdown";
import WorkspaceArt from "./WorkspaceArt";
import UiIcon from "./UiIcon";

export function parseFlashcards(raw, count) {
  const parsed = JSON.parse(raw.trim().replace(/^```(?:json)?\s*|\s*```$/gi, ""));
  const rows = Array.isArray(parsed) ? parsed : parsed.cards;
  if (!Array.isArray(rows) || !rows.length || rows.some(card => !card || typeof card.question !== "string" || !card.question.trim() || typeof card.answer !== "string" || !card.answer.trim())) throw new Error("The AI returned incomplete flashcards. Try again.");
  return rows.slice(0, count).map(card => ({ id: makeStudyId(), question: card.question.trim().slice(0, 2000), answer: card.answer.trim().slice(0, 4000) }));
}
export default function FlashcardStudio({ file, onUpdate, getFileText }) {
  const cards = file.studyCards || [], progress = file.cardProgress || {}, known = progress.known || {}, starred = progress.starred || {}, reviews = progress.review || {};
  const [filter, setFilter] = useState("all"), [query, setQuery] = useState(""), [mode, setMode] = useState("browse"), [flipped, setFlipped] = useState(false);
  const [activeId, setActiveId] = useState(progress.activeCardId || cards[0]?.id), [edited, setEdited] = useState(null), [remove, setRemove] = useState(null), [generator, setGenerator] = useState(false);
  const [count, setCount] = useState(8), [topic, setTopic] = useState(""), [replace, setReplace] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [shuffle, setShuffle] = useState([]), [reviewed, setReviewed] = useState([]), [finished, setFinished] = useState(false), [showAll, setShowAll] = useState(false);
  const pending = useRef(null), latest = useRef({ file, onUpdate }); latest.current = { file, onUpdate };
  const status = card => known[card.id] ? "mastered" : reviews[card.id] ? "learning" : "new";
  const groups = { new: cards.filter(card => status(card) === "new").length, learning: cards.filter(card => status(card) === "learning").length, mastered: cards.filter(card => status(card) === "mastered").length };
  const visible = (() => {
    const order = new Map(shuffle.map((id, index) => [id, index]));
    return cards.filter(card => (filter === "all" || (filter === "starred" ? starred[card.id] : filter === "due" ? !reviews[card.id] || reviews[card.id].due <= Date.now() : (known[card.id] ? "mastered" : reviews[card.id] ? "learning" : "new") === filter)) && `${card.question} ${card.answer}`.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => shuffle.length ? (order.get(a.id) ?? Infinity) - (order.get(b.id) ?? Infinity) : 0);
  })();
  const index = Math.max(0, visible.findIndex(card => card.id === activeId)), card = visible[index];
  useStudyActivity(file.id, "Flashcard tutor", card ? `Front: ${card.question}${flipped ? `\nBack: ${card.answer}` : "\nThe learner is viewing the front. Offer a clue before revealing the answer."}` : "The learner is preparing flashcards.");
  useEffect(() => () => pending.current?.abort(), []);
  function update(patch) { const data = latest.current; data.onUpdate({ ...data.file, ...patch }); }
  function select(next) {
    if (!visible[next]) return;
    setActiveId(visible[next].id); setFlipped(false); setFinished(false);
    update({ cardProgress: { ...progress, activeCardId: visible[next].id } });
  }
  function grade(remembered) {
    if (!card) return;
    const review = scheduleCardReview(reviews[card.id], remembered);
    update({ cardProgress: { ...progress, known: { ...known, [card.id]: remembered && (review.level >= 3 || !!known[card.id]) }, starred,
      review: { ...reviews, [card.id]: review }, activeCardId: visible[Math.min(index + 1, visible.length - 1)]?.id } });
    setReviewed(previous => [...new Set([...previous, card.id])]); setFlipped(false);
    if (index < visible.length - 1) setActiveId(visible[index + 1].id); else setFinished(true);
  }
  useEffect(() => {
    const key = event => {
      if (mode !== "study" || edited || generator || remove || event.target.closest?.("input,textarea,select,button,[contenteditable]")) return;
      if (event.key === " " || event.key === "ArrowRight" || event.key === "ArrowLeft") event.preventDefault();
      if (event.key === " ") setFlipped(value => !value);
      if (event.key === "ArrowRight") select(index + 1);
      if (event.key === "ArrowLeft") select(index - 1);
      if (event.key === "1") grade(false);
      if (event.key === "2") grade(true);
    };
    window.addEventListener("keydown", key); return () => window.removeEventListener("keydown", key);
  });
  function saveCard(event) {
    event.preventDefault();
    if (!edited.question.trim() || !edited.answer.trim()) return;
    const next = { id: edited.id || makeStudyId(), question: edited.question.trim(), answer: edited.answer.trim() };
    update({ studyCards: edited.id ? cards.map(item => item.id === next.id ? { ...item, ...next } : item) : [...cards, next] }); setEdited(null);
  }
  async function generate(event) {
    event.preventDefault(); if (busy) return;
    if (replace && cards.length && !window.confirm("Replace the existing flashcards and their review progress? Your notes and other tools stay saved.")) return;
    const controller = new AbortController(); pending.current = controller; setBusy(true); setError("");
    try {
      const source = (file.notes || await getFileText?.() || "").trim();
      if (!source && !topic.trim()) throw new Error("Add study material or enter a topic to generate flashcards.");
      const raw = await sendStudyRequest({ messages: [{ role: "system", content: `Create ${count} useful study flashcards. Return only a JSON array of objects with question and answer strings. Use the source when supplied, keep one idea per card, ${MATH_OUTPUT_GUIDE} Do not invent details about an unavailable document.` }, { role: "user", content: `Topic: ${topic.trim() || file.name}\n\nSource:\n${source.slice(0, 15000)}` }], maxTokens: Math.min(7000, count * 220 + 500) }, controller.signal);
      if (controller.signal.aborted) return;
      const next = parseFlashcards(raw, count);
      update({ studyCards: replace ? next : [...(latest.current.file.studyCards || []), ...next], ...(replace ? { cardProgress: {} } : {}) }); setActiveId(next[0].id); setFlipped(false); setGenerator(false); setFilter("all"); setFinished(false);
    } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
    finally { if (!controller.signal.aborted) setBusy(false); pending.current = null; }
  }
  const exportCards = () => downloadStudyFile(new Blob([cards.map(item => [item.question, item.answer].map(text => text.replace(/[\t\r\n]+/g, " ")).join("\t")).join("\n")], { type: "text/tab-separated-values" }), "classio-flashcards.tsv");
  return <section className="flashcard-studio"><header className="tool-heading"><div><p className="eyebrow">A LITTLE PRACTICE, OFTEN</p><h2>Flashcards</h2><p>{cards.length ? `${cards.length} cards from your material. Study at your own pace.` : "Turn important ideas into something you can remember."}</p></div><div className="studio-controls"><button className="quiet-button" onClick={() => setEdited({ question: "", answer: "" })}><UiIcon name="plus" />Add card</button><button className="quiet-button is-primary" onClick={() => { setGenerator(true); setError(""); }}><UiIcon name="generate" />Generate cards</button></div></header>
    <div className="card-category-bar" aria-label="Card categories">{["new", "learning", "mastered"].map(type => <button key={type} aria-pressed={filter === type} className={`category-${type}`} onClick={() => { setFilter(filter === type ? "all" : type); setFlipped(false); setFinished(false); }}>{groups[type]} {type[0].toUpperCase() + type.slice(1)}</button>)}</div>
    {cards.length ? <><div className="flashcard-actions"><label className="library-search"><UiIcon name="search" /><input type="search" aria-label="Search flashcards" placeholder="Find a card" value={query} onChange={event => setQuery(event.target.value)} /></label><select aria-label="Flashcard filter" value={filter} onChange={event => { setFilter(event.target.value); setFlipped(false); }}><option value="all">All cards</option><option value="due">Due for review</option><option value="new">New</option><option value="learning">Learning</option><option value="mastered">Mastered</option><option value="starred">Starred</option></select><button className="quiet-button" onClick={() => { setShuffle(previous => previous.length ? [] : cards.map(item => item.id).sort(() => Math.random() - .5)); setFlipped(false); }} aria-pressed={!!shuffle.length}><UiIcon name="shuffle" />Shuffle</button><button className="quiet-button" onClick={exportCards} title="Download flashcards"><UiIcon name="download" /></button></div>
      {card ? <div className={`flashcard-stage ${mode === "study" ? "is-studying" : ""}`}>
        {finished ? <div className="flashcard-session-result"><UiIcon name="check" size={36} /><h3>You reviewed {reviewed.length} {reviewed.length === 1 ? "card" : "cards"}</h3><p>Your review progress is saved with this material.</p><button className="quiet-button is-primary" onClick={() => { setFinished(false); setReviewed([]); select(0); }}>Review again</button></div>
          : <><div className="flashcard-stage-top"><span>{mode === "study" ? "STUDY MODE" : "CARD PREVIEW"}</span><div><button className="quiet-button" aria-label="Edit current flashcard" onClick={() => setEdited({ ...card })}><UiIcon name="edit" size={17} /></button><button className="quiet-button" aria-label={starred[card.id] ? "Unstar card" : "Star card"} aria-pressed={!!starred[card.id]} onClick={() => update({ cardProgress: { ...progress, starred: { ...starred, [card.id]: !starred[card.id] } } })}><UiIcon name="star" size={17} /></button></div></div>
            <div className="flashcard-flip-space"><div className={`flashcard-flipper ${flipped ? "is-flipped" : ""}`} role="button" tabIndex={0} aria-label={flipped ? "Flashcard answer. Flip to question" : "Flashcard question. Flip to answer"} onClick={() => setFlipped(value => !value)} onKeyDown={event => { if (event.key === " " || event.key === "Enter") { event.preventDefault(); event.stopPropagation(); setFlipped(value => !value); } }}>
              <div className="flashcard-face is-front" aria-hidden={flipped}><p className="eyebrow">QUESTION</p><StudyMarkdown text={card.question} /><span>Tap to reveal the answer</span></div><div className="flashcard-face is-back" aria-hidden={!flipped}><p className="eyebrow">ANSWER</p><StudyMarkdown text={card.answer} /><span>Tap to return to the question</span></div>
            </div></div><div className="flashcard-navigation"><button className="quiet-button" aria-label="Previous flashcard" disabled={index === 0} onClick={() => select(index - 1)}><UiIcon name="back" /></button><span>{index + 1} / {visible.length}</span><button className="quiet-button" aria-label="Next flashcard" disabled={index >= visible.length - 1} onClick={() => select(index + 1)}><UiIcon name="arrow" /></button></div>
            {mode === "study" ? <div className="flashcard-grades"><button className="quiet-button" onClick={() => grade(false)}><UiIcon name="close" />Don’t know<span>1</span></button><button className="quiet-button is-primary" onClick={() => grade(true)}><UiIcon name="check" />Know<span>2</span></button><button className="text-button" onClick={() => setMode("browse")}>Exit study</button></div> : <button className="quiet-button is-primary flashcard-start" onClick={() => { setMode("study"); setReviewed([]); setFinished(false); setFlipped(false); }}>Start studying<UiIcon name="arrow" /></button>}
          </>}
      </div> : <div className="tool-empty"><h3>No cards match this view</h3><p>Try another category or search.</p><button className="quiet-button" onClick={() => { setFilter("all"); setQuery(""); }}>Show all cards</button></div>}
      <section className="card-list"><div className="section-heading"><h3>Your cards</h3><span>{visible.length} in this view</span></div>{(showAll ? visible : visible.slice(0, 6)).map(item => <article key={item.id}><div><StudyMarkdown className="card-list-question" text={item.question} /><StudyMarkdown text={item.answer} /></div><span className={`card-status category-${status(item)}`}>{status(item)}</span><button className="quiet-button" aria-label={`Edit card: ${item.question}`} onClick={() => setEdited({ ...item })}><UiIcon name="edit" /></button><button className="quiet-button" aria-label={`Delete card: ${item.question}`} onClick={() => setRemove(item)}><UiIcon name="trash" /></button></article>)}{visible.length > 6 && <button className="text-button" onClick={() => setShowAll(value => !value)}>{showAll ? "Show fewer cards" : `Show all ${visible.length} cards`}</button>}</section>
    </> : <div className="tool-empty"><WorkspaceArt /><h3>Make your first set</h3><p>Add a card yourself or generate a set from this file. Your study progress is kept with your material.</p><button className="quiet-button" onClick={() => setEdited({ question: "", answer: "" })}>Write a flashcard</button></div>}
    {edited && <StudioDialog title={edited.id ? "Edit flashcard" : "Add flashcard"} onClose={() => setEdited(null)}><form className="simple-workspace-form" onSubmit={saveCard}><label>Question<textarea aria-label="Question" required rows={3} value={edited.question} onChange={event => setEdited({ ...edited, question: event.target.value })} maxLength={2000} /></label><label>Answer<textarea aria-label="Answer" required rows={4} value={edited.answer} onChange={event => setEdited({ ...edited, answer: event.target.value })} maxLength={4000} /></label><footer><button className="quiet-button" type="button" onClick={() => setEdited(null)}>Cancel</button><button className="quiet-button is-primary" type="submit">Save card</button></footer></form></StudioDialog>}
    {generator && <StudioDialog title="Generate flashcards" onClose={() => setGenerator(false)}><form className="simple-workspace-form" onSubmit={generate}><label>Topic<input value={topic} onChange={event => setTopic(event.target.value)} placeholder="Optional with study material" /></label><label>Number of cards<select value={count} onChange={event => setCount(Number(event.target.value))}>{[5, 8, 12, 20].map(n => <option key={n} value={n}>{n}</option>)}</select></label>{!!cards.length && <label className="check-label"><input type="checkbox" checked={replace} onChange={event => setReplace(event.target.checked)} />Replace my existing cards</label>}<p>New cards are added to this file. Existing cards and progress are kept unless you choose Replace.</p>{error && <p role="alert">{error}</p>}<footer><button type="button" className="quiet-button" onClick={() => setGenerator(false)}>Cancel</button><button type="submit" className="quiet-button is-primary" disabled={busy}>{busy ? "Generating…" : "Generate cards"}</button></footer></form></StudioDialog>}
    {remove && <StudioDialog title="Delete flashcard?" onClose={() => setRemove(null)}><p>This removes the selected card and its review progress.</p><div className="dialog-actions"><button className="quiet-button" onClick={() => setRemove(null)}>Cancel</button><button className="quiet-button is-danger" onClick={() => { const clean = map => Object.fromEntries(Object.entries(map).filter(([id]) => id !== remove.id)); update({ studyCards: cards.filter(item => item.id !== remove.id), cardProgress: { ...progress, known: clean(known), starred: clean(starred), review: clean(reviews) } }); setRemove(null); }}>Delete card</button></div></StudioDialog>}
  </section>;
}
