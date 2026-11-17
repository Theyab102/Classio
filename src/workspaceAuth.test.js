import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import { getDoc, setDoc } from "firebase/firestore";
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
  deleteField:()=>"__delete_field__", getFirestore: () => ({}), doc: jest.fn(), getDoc: jest.fn(), setDoc: jest.fn(),
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
    expect(screen.getByRole("status")).toHaveTextContent("Preparing your study workspace");
    await act(async () => { jest.advanceTimersByTime(8000); await restore; });
    expect(screen.getByRole("heading", { name: "Library" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open folder Physics", exact: true })).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Cloud sync is unavailable");
  } finally { jest.useRealTimers(); }
});

test("explains an unauthorized Firebase hostname and keeps the guest dashboard open", async () => {
  signInWithPopup.mockRejectedValue({ code: "auth/unauthorized-domain" });
  render(<App />);
  await act(async()=>{await mockAuthListener(null);});
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

test("custom wearable profiles restore separately and save without inflating the materials document",async()=>{
 const design={name:"Blue crown",kind:"hat",region:"hat",baseStyle:5,texture:"data:image/png;base64,iVBORw0KGgo="};
 const root={metadata:{fromCache:false},data:()=>({folders:[],characterAssetProfile:true,character:{name:"Saved creator",hat:5}})};
 getDoc.mockResolvedValue(root).mockResolvedValueOnce(root).mockResolvedValueOnce({data:()=>({character:{name:"Saved creator",hat:5,customHat:design,designs:[design]}})});setDoc.mockResolvedValue();
 render(<App/>);mockAuth.currentUser={uid:"creator",displayName:"Creator"};await act(async()=>{await mockAuthListener(mockAuth.currentUser);});
 fireEvent.click(screen.getByRole("button",{name:"Edit character",exact:true}));expect(screen.getByText("Wearing Blue crown",{exact:true})).toBeInTheDocument();
 await act(async()=>{fireEvent.click(screen.getByRole("button",{name:"Save changes",exact:true}));});
 expect(setDoc.mock.calls.some(([,data])=>data.character?.customHat?.name==="Blue crown")).toBe(true);
 const workspace=setDoc.mock.calls.find(([,data])=>data.characterAssetProfile===true)?.[1];expect(workspace.character.customHat).toBeUndefined();expect(workspace.character.designs).toBeUndefined();expect(workspace.character.name).toBe("Saved creator");
});

 test('initial authentication keeps one loading shell instead of flashing an empty guest library',async()=>{render(<App/>);expect(screen.queryByRole('heading',{name:'Library'})).not.toBeInTheDocument();expect(screen.getByRole('status')).toHaveTextContent('Preparing your study workspace');await act(async()=>{await mockAuthListener(null);});expect(screen.getByRole('heading',{name:'Library'})).toBeInTheDocument();expect(screen.queryByText('Preparing your study workspace…')).not.toBeInTheDocument();});

test('a pending legacy workspace publishes bounded records before removing the obsolete root copy',async()=>{
 jest.useFakeTimers();const previousEncoder=global.TextEncoder;global.TextEncoder=require('util').TextEncoder;
 try{localStorage.setItem('classio_workspace:sync-owner',JSON.stringify({folders:[{id:'notes',name:'Saved notes',files:[{id:'one',name:'Original',notes:'Keep this note'}]}],pending:true}));getDoc.mockResolvedValue({metadata:{fromCache:false},data:()=>({folders:[]})});setDoc.mockResolvedValue();render(<App/>);mockAuth.currentUser={uid:'sync-owner',displayName:'Owner'};await act(async()=>{await mockAuthListener(mockAuth.currentUser);});await act(async()=>{jest.advanceTimersByTime(500);});await waitFor(()=>expect(setDoc.mock.calls.some(([,value])=>value.folders==='__delete_field__')).toBe(true));const manifest=setDoc.mock.calls.findIndex(([,value])=>value.format==='classio-parts-v1'),cleanup=setDoc.mock.calls.findIndex(([,value])=>value.folders==='__delete_field__');expect(manifest).toBeGreaterThanOrEqual(0);expect(cleanup).toBeGreaterThan(manifest);expect(JSON.parse(localStorage.getItem('classio_workspace:sync-owner')).folders[0].files[0].notes).toBe('Keep this note');}
 finally{global.TextEncoder=previousEncoder;jest.useRealTimers();}
});
test('permission-denied keeps cached work and reports the denied profile operation and refreshed sign-in provider',async()=>{
 const account={uid:'private-retry-owner',displayName:'Owner',providerData:[{providerId:'google.com'}],getIdTokenResult:jest.fn(async()=>({signInProvider:'google.com'}))};
 localStorage.setItem('classio_workspace:'+account.uid,JSON.stringify({folders:[{id:'saved',name:'My saved chapter',files:[{id:'one',name:'Original note',notes:'Keep this writing'}]}],pending:true}));
 getDoc.mockRejectedValue(Object.assign(new Error('Denied'),{code:'permission-denied'}));
 render(<App/>);mockAuth.currentUser=account;await act(async()=>{await mockAuthListener(account);});
 expect(screen.getAllByText('My saved chapter').length).toBeGreaterThan(0);expect(setDoc).not.toHaveBeenCalled();
 expect(screen.getByText(/^Cloud access was denied/)).toBeInTheDocument();
 fireEvent.click(screen.getByRole('button',{name:'Retry sync',exact:true}));await waitFor(()=>expect(account.getIdTokenResult).toHaveBeenCalledWith(true));
 await waitFor(()=>expect(screen.getByText('refreshed ID token')).toBeInTheDocument());
 expect(screen.getByText('read profile',{exact:true})).toBeInTheDocument();expect(screen.getByText('users/{uid}',{exact:true})).toBeInTheDocument();expect(screen.getByText('classio-4378f',{exact:true})).toBeInTheDocument();
 expect(document.querySelector('.cloud-sync-details').textContent).not.toContain(account.uid);expect(localStorage.getItem('classio_workspace:'+account.uid)).toContain('Keep this writing');
});
