import { sanitizeNoteHtml } from "./noteFormat";

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
test('table column widths and image provenance survive restoration without admitting arbitrary attributes',()=>{
  const html=sanitizeNoteHtml('<table><tr><td colwidth="120,180"><p><strong><em><span style="font-family:Fraunces Variable;font-size:24px">Cell</span></em></strong></p></td></tr></table><img src="https://example.com/manual.png" data-classio-image-origin="manual" data-classio-image-query="trajectory" data-unsafe="x">');
  expect(html).toContain('colwidth="120,180"');expect(html).toContain('data-classio-image-origin="manual"');expect(html).not.toContain('data-unsafe');expect(html).toContain('font-size: 24px');
});
