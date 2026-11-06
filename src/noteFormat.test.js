import { sanitizeNoteHtml } from "./noteFormat";

jest.mock("marked", () => ({ marked: { parse: jest.fn() } }));
jest.mock("./pdf", () => ({ openPdf: jest.fn() }));

test("notes retain supported font formatting while stripping executable and arbitrary styles", () => {
  const html = sanitizeNoteHtml(`<p style="text-align:center"><span style="font-family:'Fraunces Variable';font-size:20px;position:fixed;background:url(https://example.com/track)">Study notes</span></p><script>alert(1)</script>`);
  expect(html).toContain("text-align: center");
  expect(html).toContain("Fraunces Variable");
  expect(html).toContain("font-size: 20px");
  expect(html).not.toMatch(/position|background|example.com|<script/);
  const rejected = sanitizeNoteHtml('<span style="font-family:RemoteFont;font-size:500px">Notes</span>');
  expect(rejected).not.toMatch(/RemoteFont|500px/);
});
