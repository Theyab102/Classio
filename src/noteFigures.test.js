import { enrichNoteImages } from "./noteFigures";
import { notesWithSourceImages } from "./linkSource";
import { markdownToNoteHtml } from "./noteFormat";
import { likelyStudyImage, searchStudyImages } from "./documentMedia";
jest.mock("./pdf", () => ({ openPdf: jest.fn(), loadPdfLibrary: jest.fn() }));

const notes = "# Physics\n\n## Projectile motion\n\nA projectile follows a curved trajectory because gravity changes its vertical velocity.\n\n## Terminal velocity\n\nDrag balances weight so a falling object stops accelerating.\n\n## Suggested Visual Searches\n\n[[IMAGE:projectile trajectory]]\n[[IMAGE:terminal velocity]]";
test("grouped image markers become real figures beside their related explanations", async () => {
  const search = jest.fn(query => Promise.resolve([{ src: "https://example.com/" + (query.startsWith("projectile") ? "projectile" : "terminal") + ".png", title: query, context: "Physics, terminal velocity and drag forces.", source: "https://example.com/source", credit: "Source credit" }]));
  const { html, added } = await enrichNoteImages(notes, { useWeb: true, search });
  const root = document.createElement("div"); root.innerHTML = html;
  expect(added).toBe(2); expect(html).not.toContain("[[IMAGE:"); expect(html).not.toContain("Suggested Visual Searches");
  const headings = [...root.querySelectorAll("h2")];
  expect(headings[0].nextElementSibling.nextElementSibling.getAttribute("src")).toBe("https://example.com/projectile.png");
  expect(headings[1].nextElementSibling.nextElementSibling.getAttribute("src")).toBe("https://example.com/terminal.png");
});
test("failed image searches leave readable notes without raw placeholders", async () => {
  const { html, missing } = await enrichNoteImages(notes, { useWeb: true, search: async () => { throw new Error("unavailable"); } });
  expect(missing).toBe(2); expect(html).toContain("Drag balances weight"); expect(html).not.toContain("[[IMAGE:"); expect(html).not.toContain("Suggested Visual Searches");
});
test("webpage source images also move out of a trailing image list into the matching section", () => {
  const source = { url: "https://example.com/physics", images: [{ src: "https://example.com/projectile.png", title: "Projectile trajectory" }] };
  const html = markdownToNoteHtml(notes.replace(/\[\[IMAGE:[^\]]+\]\]/g, "[[SOURCE_IMAGE:1]]"));
  const output = notesWithSourceImages(notes.replace(/\[\[IMAGE:[^\]]+\]\]/g, "[[SOURCE_IMAGE:1]]"), source);
  const root = document.createElement("div"); root.innerHTML = markdownToNoteHtml(output);
  expect(html).toContain("SOURCE_IMAGE");
  expect(root.querySelector("h2").nextElementSibling.nextElementSibling.querySelector("img").src).toBe("https://example.com/projectile.png");
  expect(root.querySelectorAll("img")).toHaveLength(1); expect(output).not.toContain("SOURCE_IMAGE");
});
test("licensed vector diagrams use their rendered thumbnails and actual image descriptions for relevance", async () => {
  const previousFetch = global.fetch;
  global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({ query: { pages: { 1: { index: 1, title: "File:Projectile motion.svg", imageinfo: [{ mime: "image/svg+xml", thumburl: "https://upload.wikimedia.org/projectile.svg.png", descriptionurl: "https://commons.wikimedia.org/wiki/File:Projectile_motion.svg", extmetadata: { LicenseShortName: { value: "CC BY-SA" }, Artist: { value: "A credited illustrator" }, ImageDescription: { value: "A parabolic projectile trajectory under gravity." } } }] } } } }) }));
  try { const results = await searchStudyImages("parabolic projectile trajectory"); expect(likelyStudyImage("parabolic projectile trajectory", results).src).toBe("https://upload.wikimedia.org/projectile.svg.png"); expect(results[0].credit).toContain("CC BY-SA"); }
  finally { global.fetch = previousFetch; }
});
test("existing suggestions can be repaired without dropping formatting or existing figures", async () => {
  const existingHtml = '<h2>Projectile motion</h2><p style="text-align:center"><strong style="font-size:20px">A projectile follows a curved trajectory under gravity.</strong></p><p>[[IMAGE:projectile trajectory]]</p><img src="https://example.com/original.png" alt="Existing illustration">';
  const { html } = await enrichNoteImages("", { existingHtml, useWeb: true, search: async () => [{ src: "https://example.com/trajectory.png", title: "Projectile trajectory" }] });
  expect(html).toContain('text-align: center'); expect(html).toContain('font-size: 20px'); expect(html).toContain('<strong');
  expect(html).toContain('src="https://example.com/original.png"'); expect(html).toContain('src="https://example.com/trajectory.png"'); expect(html).not.toContain("[[IMAGE:");
});
