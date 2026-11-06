const paths = {
  notes: "M5 3h10l4 4v14H5z M14 3v5h5 M8 12h8M8 16h6",
  folder: "M3 6h7l2 2h9v12H3z",
  plus: "M12 5v14M5 12h14",
  more: "M5 12h.01M12 12h.01M19 12h.01",
  close: "M6 6l12 12M6 18L18 6",
  chevron: "m9 5 7 7-7 7",
  back: "M19 12H5m7-7-7 7 7 7",
  bold: "M7 4h6a4 4 0 0 1 0 8H7zm0 8h7a4 4 0 0 1 0 8H7z",
  italic: "M10 4h9M5 20h9M15 4 9 20",
  underline: "M6 4v7a6 6 0 0 0 12 0V4M4 21h16",
  list: "M9 6h12M9 12h12M9 18h12M3 6h.01M3 12h.01M3 18h.01",
  table: "M3 4h18v16H3zM3 10h18M9 4v16M15 10v10",
  image: "M3 3h18v18H3z m0 13 5-5 5 5 3-3 5 5M15 7h.01",
  undo: "M8 4 3 9l5 5M3 9h11a7 7 0 0 1 0 14",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
  generate: "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z",
  source: "M4 3h16v18H4zM8 7h8M8 11h8M8 15h5",
  cards: "M3 3h14v14H3zM7 21h14V7",
  podcast: "M4 14v-3a8 8 0 0 1 16 0v3M4 12h3v8H4zM17 12h3v8h-3z",
  slides: "M3 3h18v13H3zM12 16v5M7 21h10",
  mic: "M9 4a3 3 0 0 1 6 0v8a3 3 0 0 1-6 0zM5 10v2a7 7 0 0 0 14 0v-2M12 19v3",
  link: "m10 14 4-4M8 16l-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0M16 8l2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0",
  quiz: "M4 3h16v18H4z m4 9 3 3 6-7",
};

export default function UiIcon({ name, size = 19, ...props }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name] || paths.notes} /></svg>;
}
