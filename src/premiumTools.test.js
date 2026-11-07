import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import QuizStudio, { parseQuiz } from "./QuizStudio";
import FlashcardStudio, { parseFlashcards } from "./FlashcardStudio";
import AvatarEditor from "./AvatarEditor";
import LibraryHome from "./LibraryHome";
import { sendStudyRequest } from "./studyTools";
import { parseEpisode, pcmWave } from "./podcastAudio";

jest.mock("./pdf", () => ({ openPdf: jest.fn(), loadPdfLibrary: jest.fn() }));
jest.mock("./studyTools", () => ({ ...jest.requireActual("./studyTools"), sendStudyRequest: jest.fn() }));
const question = { question: "Which unit measures force?", options: ["Newton", "Joule", "Watt", "Volt"], correct: 0, hint: "Think of the unit in F = ma.", explanation: "Force is measured in newtons." };
function Harness({ Component, initial, changed }) {
  const [file, setFile] = useState(initial);
  return <Component file={file} getFileText={async () => "Forces change motion."} onUpdate={next => { changed(next); setFile(next); }} />;
}

test("quiz hints, checked answers and navigation preserve the actual attempt", () => {
  const changed = jest.fn();
  render(<Harness Component={QuizStudio} changed={changed} initial={{ id: "physics", name: "Forces", notes: "Force is measured in newtons.", quiz: { questions: [question, { ...question, question: "Name the SI unit of force." }], answers: {}, checked: {}, active: 0 } }} />);
  fireEvent.click(screen.getByRole("button", { name: "Show hint" }));
  expect(screen.getByText(question.hint)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /A Newton/ }));
  fireEvent.click(screen.getByRole("button", { name: "Check answer", exact: true }));
  expect(changed.mock.calls.at(-1)[0].quiz.checked[0]).toBe(true);
  expect(changed.mock.calls.at(-1)[0].quiz.answers[0]).toBe(0);
  fireEvent.click(screen.getByRole("button", { name: "Next", exact: true }));
  expect(changed.mock.calls.at(-1)[0].quiz.active).toBe(1);
  fireEvent.click(screen.getByRole("button", { name: "Previous", exact: true }));
  expect(screen.getByText("Correct.")).toBeInTheDocument();
});

test("editing and adding flashcards keep existing cards and review progress", () => {
  const changed = jest.fn();
  render(<Harness Component={FlashcardStudio} changed={changed} initial={{ id: "physics", name: "Forces", studyCards: [{ id: "one", question: "Force?", answer: "A push or pull." }], cardProgress: { known: { one: true }, review: { one: { level: 3, due: 1 } } } }} />);
  fireEvent.click(screen.getByRole("button", { name: "Edit current flashcard" }));
  fireEvent.change(screen.getByLabelText("Question", { exact: true }), { target: { value: "What changes motion?" } });
  fireEvent.change(screen.getByLabelText("Answer", { exact: true }), { target: { value: "A net force." } });
  fireEvent.click(screen.getByRole("button", { name: "Save card" }));
  expect(changed.mock.calls.at(-1)[0].studyCards[0]).toEqual({ id: "one", question: "What changes motion?", answer: "A net force." });
  fireEvent.click(screen.getByRole("button", { name: "Add card", exact: true }));
  fireEvent.change(screen.getByLabelText("Question", { exact: true }), { target: { value: "Unit of force?" } });
  fireEvent.change(screen.getByLabelText("Answer", { exact: true }), { target: { value: "Newton." } });
  fireEvent.click(screen.getByRole("button", { name: "Save card" }));
  const next = changed.mock.calls.at(-1)[0];
  expect(next.studyCards).toHaveLength(2); expect(next.cardProgress.known.one).toBe(true); expect(next.cardProgress.review.one.level).toBe(3);
});

test("flashcard generation appends real returned cards without clearing progress", async () => {
  sendStudyRequest.mockResolvedValue(JSON.stringify([{ question: "What is acceleration?", answer: "Rate of change of velocity." }]));
  const changed = jest.fn();
  render(<Harness Component={FlashcardStudio} changed={changed} initial={{ id: "p", name: "Motion", notes: "Acceleration is a change in velocity per second.", studyCards: [{ id: "old", question: "Speed?", answer: "Distance per time." }], cardProgress: { starred: { old: true } } }} />);
  fireEvent.click(screen.getByRole("button", { name: "Generate cards", exact: true }));
  fireEvent.submit(screen.getByRole("dialog").querySelector("form"));
  await waitFor(() => expect(changed).toHaveBeenCalled());
  const next = changed.mock.calls.at(-1)[0]; expect(next.studyCards).toHaveLength(2); expect(next.cardProgress.starred.old).toBe(true);
  expect(sendStudyRequest.mock.calls[0][0].messages[1].content).toContain("Acceleration is a change in velocity");
});

test("avatar choices remain a draft until saved and Cancel leaves the original intact", () => {
  const changed = jest.fn(), close = jest.fn(), original = { name: "Sam", hairStyle: 0, skin: "#c68642" };
  render(<AvatarEditor character={original} onChange={changed} onClose={close} />);
  fireEvent.click(screen.getByRole("button", { name: "Hair", exact: true }));
  fireEvent.click(screen.getByRole("button", { name: "Hairstyle: Two buns", exact: true }));
  expect(changed).not.toHaveBeenCalled(); expect(original.hairStyle).toBe(0);
  fireEvent.click(screen.getByRole("button", { name: "Cancel", exact: true }));
  expect(close).toHaveBeenCalledTimes(1); expect(changed).not.toHaveBeenCalled();
});

test("the four restored home shortcuts call the preserved feature handlers", () => {
  const createNote = jest.fn(), upload = jest.fn(), record = jest.fn(), addLink = jest.fn();
  render(<LibraryHome folders={[]} createNote={createNote} upload={upload} record={record} addLink={addLink} />);
  for (const name of ["Blank document", "Upload document", "Record audio", "Website link"]) fireEvent.click(screen.getByRole("button", { name, exact: true }));
  expect(createNote).toHaveBeenCalledWith("write"); expect(upload).toHaveBeenCalledTimes(1); expect(record).toHaveBeenCalledTimes(1); expect(addLink).toHaveBeenCalledTimes(1);
});

test("generated structures reject incomplete outputs and audio timestamps derive from real PCM frames", () => {
  expect(parseQuiz(JSON.stringify([question]))[0].correct).toBe(0);
  expect(() => parseQuiz(JSON.stringify([{ ...question, correct: 6 }]))).toThrow();
  expect(() => parseFlashcards('[{"question":"Missing answer"}]', 8)).toThrow();
  const turns = Array.from({ length: 4 }, (_, i) => ({ speaker: i % 2 ? "B" : "A", section: "Forces", text: "A force changes motion." }));
  expect(parseEpisode(JSON.stringify({ title: "Forces", turns })).turns).toHaveLength(4);
  expect(() => parseEpisode(JSON.stringify({ title: "Forces", turns: turns.map(turn => ({ ...turn, speaker: "A" })) }))).toThrow();
  const blob = pcmWave([new Float32Array(22050)], 22050); expect(blob.type).toBe("audio/wav"); expect(blob.size).toBe(44 + 22050 * 2);
});
