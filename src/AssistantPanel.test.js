import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import AssistantPanel from "./AssistantPanel";
import { sendStudyRequest } from "./studyTools";
import { parseStudyRequest } from "../server/study-request";

jest.mock("./pdf", () => ({ openPdf: jest.fn(), loadPdfLibrary: jest.fn() }));
jest.mock("./studyTools", () => ({ ...jest.requireActual("./studyTools"), sendStudyRequest: jest.fn() }));

test("retrying a failed image question keeps its attachment in a valid server request", async () => {
  window.matchMedia = jest.fn(() => ({ matches: false, addEventListener: jest.fn(), removeEventListener: jest.fn() }));
  HTMLElement.prototype.scrollIntoView = jest.fn();
  const image = "data:image/png;base64,aGVsbG8=";
  function Harness() {
    const [messages, setMessages] = useState([{ role: "user", content: "Explain this diagram", image, failed: true }]);
    const [pending, setPending] = useState(false), [draft, setDraft] = useState("");
    return <AssistantPanel context={{ id: "forces", name: "Forces", notes: "Force is measured in newtons." }} messages={messages} onMessagesChange={setMessages} pending={pending} onPendingChange={setPending} draft={draft} onDraftChange={setDraft} onClose={() => {}} />;
  }
  sendStudyRequest.mockResolvedValue("The diagram shows a balanced force.");
  render(<Harness />);
  expect(screen.getAllByRole("button", { name: "Retry response", exact: true })).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Retry response", exact: true }));
  await screen.findByText("The diagram shows a balanced force.");
  const request = sendStudyRequest.mock.calls[0][0];
  expect(parseStudyRequest(request)?.vision).toBe(true);
  expect(request.messages.at(-1).content[0].image_url.url).toBe(image);
  expect(request.messages[0].content).toContain("Force is measured in newtons.");
});
