import { lockBodyScroll } from "./bodyScrollLock";
import { notesWithSourceImages, sourceTables } from "./linkSource";
import { spokenTurn, parseEpisode } from "./podcastAudio";
import { markdownToNoteHtml } from "./noteFormat";
jest.mock("./pdf", () => ({ openPdf: jest.fn() }));

test("overlapping drawers release scrolling in either close order", () => {
  document.body.style.overflow = "auto";
  const sidebar = lockBodyScroll(), avatar = lockBodyScroll();
  expect(document.documentElement.classList.contains("classio-scroll-locked")).toBe(true);
  expect(document.body.style.overflow).toBe("hidden"); sidebar();
  expect(document.body.style.overflow).toBe("hidden"); avatar();
  expect(document.body.style.overflow).toBe("auto"); avatar();
  expect(document.documentElement.classList.contains("classio-scroll-locked")).toBe(false);
  expect(document.body.style.overflow).toBe("auto");
  document.body.style.overflow = "";
});
test("podcast speech strips metadata labels but keeps names within natural sentences", () => {
  expect(spokenTurn("Mira: Jules: A force changes motion.")).toBe("A force changes motion.");
  expect(spokenTurn("Jules, can you explain gravity?")).toBe("Jules, can you explain gravity?");
  const turns = Array.from({ length: 4 }, (_, i) => ({ speaker: i % 2 ? "B" : "A", section: "Physics", text: `${i % 2 ? "Jules" : "Mira"}: Gravity changes vertical velocity.` }));
  expect(parseEpisode(JSON.stringify({ title: "Motion", turns })).turns.every(turn => !/Mira:|Jules:/.test(turn.text))).toBe(true);
});
test("link notes render headings, emphasis, tables and only genuine source figures", () => {
  const source = { url: "https://example.com/physics", images: [{ src: "https://example.com/trajectory.png", title: "Projectile trajectory" }] };
  const raw = "# Motion\n\n**Key idea:** horizontal velocity is constant.\n\n" + sourceTables([[['Quantity', 'Unit'], ['Speed', 'm/s']]]) + '\n\n![Invented figure](https://invented.example/figure.png)\n<img src="https://invented.example/other.png">\n\n[[SOURCE_IMAGE:1]]';
  const html = markdownToNoteHtml(notesWithSourceImages(raw, source));
  expect(html).toContain("<h1>Motion</h1>"); expect(html).toContain("<strong>Key idea:</strong>"); expect(html).toContain("<table>");
  expect(html).toContain('src="https://example.com/trajectory.png"'); expect(html).not.toContain("SOURCE_IMAGE");
  expect(html).not.toContain("invented.example");
});
