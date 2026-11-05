import { fireEvent, render, screen } from "@testing-library/react";
import { DeckDesignPicker, deckDesigns, normalizeSlideTable, parseDeckResponse } from "./DeckDesign";
import { parsePowerPointTables } from "./PowerPointViewer";
import { pdfZoomScale } from "./LocalPdfViewer";

jest.mock("./pdf", () => ({ openPdf: jest.fn(), loadPdfLibrary: jest.fn() }));

test("PowerPoint tables preserve real cells, spans, and paragraph breaks", () => {
  const xml = new DOMParser().parseFromString(`<a:tbl xmlns:a="urn:drawing">
    <a:tblPr firstRow="1"/><a:tblGrid><a:gridCol w="100"/><a:gridCol w="200"/></a:tblGrid>
    <a:tr><a:tc gridSpan="2"><a:txBody><a:p><a:r><a:t>Motion</a:t></a:r></a:p><a:p><a:r><a:t>Units</a:t></a:r></a:p></a:txBody></a:tc><a:tc hMerge="1"/></a:tr>
    <a:tr><a:tc><a:p><a:t>Force</a:t></a:p></a:tc><a:tc><a:p><a:t>Newton</a:t></a:p></a:tc></a:tr>
  </a:tbl>`, "application/xml");
  const [table] = parsePowerPointTables(xml);
  expect(table.columns).toEqual([100, 200]);
  expect(table.header).toBe(true);
  expect(table.rows[0]).toEqual([{ text: "Motion\nUnits", colSpan: 2, rowSpan: 1, fill: undefined }]);
  expect(table.rows[1].map(cell => cell.text)).toEqual(["Force", "Newton"]);
});

test("PDF fit uses a stable reference and explicit zoom is independent of the view width", () => {
  expect(pdfZoomScale("fit", 720, 600)).toBe(1.2);
  expect(pdfZoomScale("150", 500, 600)).toBe(pdfZoomScale("150", 900, 760));
  expect(pdfZoomScale("150", 900, 760)).toBe(2);
});

test("AI presentation data keeps rectangular tables and offers distinct selectable layouts", () => {
  expect(normalizeSlideTable([["Name", "Unit"], ["Force"]])).toEqual([["Name", "Unit"], ["Force", ""]]);
  const deck = parseDeckResponse(JSON.stringify({ title:"Physics", slides: [
    { type:"title", title:"Motion" }, { type:"table", title:"Compare", table:[["Force","Unit"],["Weight","Newton"]] },
  ] }), 3);
  expect(deck.slides[1].table[1]).toEqual(["Weight", "Newton"]);
  const themes = deckDesigns.map(item => ({ id:item.theme, bg:"#ffffff", text:"#000000", accent:"#16564c", sub:"#222222" }));
  const onChoose = jest.fn();
  render(<DeckDesignPicker layouts={deckDesigns} deck={deck} themes={themes} onChoose={onChoose} onCancel={() => {}} />);
  expect(screen.getAllByRole("table")).toHaveLength(3);
  fireEvent.click(screen.getByRole("button", { name:"Use Editorial" }));
  expect(onChoose).toHaveBeenCalledWith(deckDesigns[1]);
});
