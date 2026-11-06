import { act, fireEvent, render, screen } from "@testing-library/react";
import Entry from "./Entry";
import { checkBirthDate } from "./eligibility";
import { escapeHtml, sanitizeDocumentHtml } from "./safeContent";
import { parseStudyRequest } from "../server/study-request";

const mockWorkspace = jest.fn(() => <p>Workspace loaded</p>);
jest.mock("./App", () => ({ __esModule: true, default: () => mockWorkspace() }));

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  window.history.replaceState({}, "", "/");
  mockWorkspace.mockImplementation(() => <p>Workspace loaded</p>);
});

test("remembers accepted eligibility when a returning visitor opens a new tab", async () => {
  const page = render(<Entry />);
  enterBirthDate("2000-01-01");
  fireEvent.click(screen.getByRole("checkbox"));
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Open workspace" })); });
  expect(await screen.findByText("Workspace loaded")).toBeInTheDocument();
  page.unmount();
  sessionStorage.clear();
  render(<Entry />);
  expect(await screen.findByText("Workspace loaded")).toBeInTheDocument();
  expect(screen.queryByLabelText("Date of birth")).not.toBeInTheDocument();
  expect(localStorage.getItem("classio_access_v2")).not.toContain("2000-01-01");
});

function enterBirthDate(value) {
  fireEvent.click(screen.getByRole("button", { name: "Get started" }));
  const field = screen.getByLabelText("Date of birth");
  fireEvent.change(field, { target: { value } });
  fireEvent.submit(field.form);
}

test("blocks a younger visitor before loading the workspace, including after reload", () => {
  const page = render(<Entry />);
  enterBirthDate("2020-01-01");
  expect(screen.getByRole("heading", { name: "Classio is unavailable to you" })).toBeInTheDocument();
  expect(sessionStorage.getItem("classio_access_v1")).toBe("blocked");
  expect(mockWorkspace).not.toHaveBeenCalled();
  page.unmount();
  render(<Entry />);
  expect(screen.getByRole("heading", { name: "Classio is unavailable to you" })).toBeInTheDocument();
});

test("loads the workspace only after the terms step for an eligible visitor", async () => {
  render(<Entry />);
  enterBirthDate("2000-01-01");
  expect(mockWorkspace).not.toHaveBeenCalled();
  expect(sessionStorage.getItem("classio_access_v1")).toBe("eligible");
  fireEvent.click(screen.getByRole("checkbox"));
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Open workspace" })); });
  expect(await screen.findByText("Workspace loaded")).toBeInTheDocument();
  expect(sessionStorage.getItem("classio_access_v1")).toBe("accepted");
});

test.each([
  ["/privacy", "Privacy Policy"],
  ["/terms", "Terms and Conditions"],
  ["/copyright", "Copyright"],
])("opens %s without starting the workspace", (url, title) => {
  window.history.replaceState({}, "", url);
  render(<Entry />);
  expect(screen.getByRole("heading", { name: title })).toBeInTheDocument();
  expect(mockWorkspace).not.toHaveBeenCalled();
});

test("checks the exact 13th birthday and rejects invalid or future dates", () => {
  const today = new Date(2026, 10, 5);
  expect(checkBirthDate("2013-11-05", today)).toBe(true);
  expect(checkBirthDate("2013-11-06", today)).toBe(false);
  expect(checkBirthDate("2026-02-30", today)).toBeNull();
  expect(checkBirthDate("2026-11-06", today)).toBeNull();
  expect(checkBirthDate("2011-03-01", new Date(2024, 1, 29))).toBe(false);
});

test("escapes slide text and strips executable content from documents", () => {
  expect(escapeHtml('<img src=x onerror="alert(1)">')).toBe("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;");
  const html = sanitizeDocumentHtml('<p>Notes</p><a href="javascript:alert(1)">Link</a><script>alert(1)</script>');
  expect(html).toContain("<p>Notes</p>");
  expect(html).not.toMatch(/javascript:|<script/);
});

test("only accepts the supported message formats and refuses remote image URLs", () => {
  const text = { messages: [{ role: "user", content: "Explain gravity" }], maxTokens: 500 };
  expect(parseStudyRequest(text)).toEqual({ ...text, vision: false });
  expect(parseStudyRequest({ ...text, maxTokens: 90_000 })).toBeNull();

  const image = url => ({ messages: [{ role: "user", content: [
    { type: "image_url", image_url: { url } }, { type: "text", text: "Explain this." },
  ] }], maxTokens: 500 });
  expect(parseStudyRequest(image("data:image/png;base64,aGVsbG8="))?.vision).toBe(true);
  expect(parseStudyRequest(image("https://example.com/image.png"))).toBeNull();
});

test("guest AI requests send no account token", async () => {
  const { sendStudyRequest } = require("./studyTools");
  const originalFetch = global.fetch;
  global.fetch = jest.fn(async () => ({ ok: true, json: async () => ({ text: "Momentum is mass times velocity." }) }));
  try {
    const payload = { messages: [{ role: "user", content: "Explain momentum." }], maxTokens: 500 };
    expect(await sendStudyRequest(payload)).toMatch(/momentum/i);
    expect(global.fetch).toHaveBeenCalledWith("/api/study", expect.objectContaining({
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    }));
  } finally { global.fetch = originalFetch; }
});

test("session import combines the guest Inbox with the saved Inbox", () => {
  const { mergeSessionFolders } = require("./studyTools");
  const saved = [{ id: "inbox", name: "Saved Inbox", files: [{ id: "old" }] }];
  const session = [{ id: "inbox", name: "Inbox", files: [{ id: "new", notes: "Guest notes" }] }];
  expect(mergeSessionFolders(saved, session)).toEqual([
    { id: "inbox", name: "Saved Inbox", files: [{ id: "old" }, { id: "new", notes: "Guest notes" }] },
  ]);
  expect(saved[0].files).toEqual([{ id: "old" }]);
});

test("missed cards return sooner and successful reviews get a longer interval", () => {
  const { scheduleCardReview } = require("./studyTools");
  expect(scheduleCardReview({ level: 3 }, false, 1000)).toEqual({ level: 0, due: 601000 });
  const first = scheduleCardReview(undefined, true, 1000);
  expect(first).toEqual({ level: 1, due: 86401000 });
  expect(scheduleCardReview(first, true, 1000)).toEqual({ level: 2, due: 259201000 });
});

test("course generation rejects invalid quiz answers", () => {
  const { parseLessonPlan } = require("./LearnMode");
  const unit = { title: "Forces", lesson: "Forces change motion.", question: "Unit of force?",
    options: ["Joule", "Newton", "Watt"], correct: 1, explanation: "Force is measured in newtons." };
  expect(parseLessonPlan(JSON.stringify([unit, unit]))).toEqual([unit, unit]);
  expect(() => parseLessonPlan(JSON.stringify([unit, { ...unit, correct: 3 }]))).toThrow();
});
