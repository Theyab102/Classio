import { act, fireEvent, render, screen } from "@testing-library/react";
import App from "./App";
import { getDoc } from "firebase/firestore";
import { getAuth, signInWithPopup } from "firebase/auth";

const mockAuth = getAuth();
let mockAuthListener;
jest.mock("firebase/app", () => ({ initializeApp: () => ({}) }));
jest.mock("firebase/auth", () => {
  const auth = { currentUser: null };
  return {
    getAuth: () => auth,
    GoogleAuthProvider: jest.fn(),
    signInWithPopup: jest.fn(),
    signOut: jest.fn(),
    onAuthStateChanged: (_auth, listener) => { mockAuthListener = listener; return jest.fn(); },
  };
});
jest.mock("firebase/firestore", () => ({
  getFirestore: () => ({}), doc: jest.fn(), getDoc: jest.fn(), setDoc: jest.fn(),
}));
jest.mock("./pdf", () => ({ openPdf: jest.fn(), loadPdfLibrary: jest.fn() }));
jest.mock("./NotesEditor", () => () => null);
jest.mock("./pptxExport", () => ({ loadPptxExporter: jest.fn() }));

beforeEach(() => {
  window.matchMedia = jest.fn(() => ({ matches: false, addEventListener: jest.fn(), removeEventListener: jest.fn() }));
  HTMLElement.prototype.scrollIntoView = jest.fn();
  localStorage.clear();
  localStorage.setItem("classio_onboarded", "1");
  mockAuth.currentUser = null;
  jest.clearAllMocks();
});

test("opens a signed-in cached workspace when Firebase never answers", async () => {
  jest.useFakeTimers();
  try {
    localStorage.setItem("classio_workspace:account-a", JSON.stringify({
      folders: [{ id: "physics", name: "Physics", files: [] }],
    }));
    getDoc.mockImplementation(() => new Promise(() => {}));
    render(<App />);
    mockAuth.currentUser = { uid: "account-a", displayName: "Student" };
    let restore;
    await act(async () => { restore = mockAuthListener(mockAuth.currentUser); });
    expect(screen.getByRole("progressbar", { name: "Preparing workspace" })).toBeInTheDocument();
    await act(async () => { jest.advanceTimersByTime(8000); await restore; });
    expect(screen.getByRole("heading", { name: "Library" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open folder Physics", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Cloud sync is unavailable");
  } finally { jest.useRealTimers(); }
});

test("explains an unauthorized Firebase hostname and keeps the guest dashboard open", async () => {
  signInWithPopup.mockRejectedValue({ code: "auth/unauthorized-domain" });
  render(<App />);
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Sign in to save" })); });
  expect(screen.getByRole("alert")).toHaveTextContent("authorized domains");
  expect(screen.getByRole("heading", { name: "Library" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Sign in to save" })).toBeEnabled();
});

test("switching accounts isolates saved characters and assistant conversations", async () => {
  localStorage.setItem("classio_chat:account-a", JSON.stringify([{ role: "user", content: "Account A private conversation" }]));
  getDoc.mockResolvedValue({ metadata: { fromCache: false }, data: () => ({ folders: [], character: { name: "Account A character", hairStyle: 13 } }) });
  render(<App />);
  mockAuth.currentUser = { uid: "account-a", displayName: "Account A" };
  await act(async () => { await mockAuthListener(mockAuth.currentUser); });
  fireEvent.click(screen.getByRole("button", { name: "AI Assistant", exact: true }));
  expect(screen.getByText("Account A private conversation")).toBeInTheDocument();
  getDoc.mockResolvedValue({ metadata: { fromCache: false }, data: () => ({ folders: [] }) });
  mockAuth.currentUser = { uid: "account-b", displayName: "Account B" };
  await act(async () => { await mockAuthListener(mockAuth.currentUser); });
  fireEvent.click(screen.getByRole("button", { name: "AI Assistant", exact: true }));
  expect(screen.queryByText("Account A private conversation")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Close AI Assistant", exact: true }));
  fireEvent.click(screen.getByRole("button", { name: "Edit character", exact: true }));
  expect(screen.getByLabelText("Display name", { exact: true })).toHaveValue("");
  expect(JSON.parse(localStorage.getItem("classio_char:account-a")).name).toBe("Account A character");
});

test("opens saved notes when the browser file database is blocked", async () => {
  jest.useFakeTimers();
  const originalDatabase = Object.getOwnPropertyDescriptor(global, "indexedDB");
  Object.defineProperty(global, "indexedDB", { configurable: true, value: { open: () => ({}) } });
  try {
    getDoc.mockResolvedValue({ metadata: { fromCache: false }, data: () => ({
      folders: [{ id: "physics", name: "Physics", files: [{ id: "notes", name: "Forces", notes: "F = ma" }] }],
    }) });
    render(<App />);
    mockAuth.currentUser = { uid: "account-a", displayName: "Student" };
    let restore;
    await act(async () => { restore = mockAuthListener(mockAuth.currentUser); });
    await act(async () => { jest.advanceTimersByTime(8000); await restore; });
    expect(screen.getByRole("heading", { name: "Library" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open folder Physics", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Original uploads could not be restored");
  } finally {
    if (originalDatabase) Object.defineProperty(global, "indexedDB", originalDatabase);
    else delete global.indexedDB;
    jest.useRealTimers();
  }
});
