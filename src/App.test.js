import { act, fireEvent, render, screen } from "@testing-library/react";
import Entry from "./Entry";
import { checkBirthDate } from "./eligibility";
import { escapeHtml, sanitizeDocumentHtml } from "./safeContent";
import { restoreSignedInAccount } from "./firebase";
import { waitFor } from "@testing-library/react";
import { parseStudyRequest } from "../server/study-request";
import { setLanguages } from './locale';

jest.mock('./firebase', () => ({ auth: { currentUser: null }, accountHeaders: jest.fn(async () => ({})), restoreSignedInAccount:jest.fn() }));
let mockAccessStage = 'new';
const mockWorkspace = jest.fn(() => <p>Workspace loaded</p>);
jest.mock("./App", () => ({ __esModule: true, default: () => mockWorkspace() }));

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  setLanguages({ interfaceLanguage: 'en', replyLanguage: 'auto' }, { remote: true });
  mockAccessStage = 'new';
  restoreSignedInAccount.mockReset();
  global.fetch = jest.fn(async (_url, options) => {
    if (options?.method === 'POST') {
      const body = JSON.parse(options.body);
      if (body.action === 'screen') mockAccessStage = checkBirthDate(body.birthDate) ? 'eligible' : 'blocked';
      if (body.action === 'accept' || body.action==='restore') mockAccessStage = 'accepted';
    }
    return { ok: true, json: async () => ({ stage: mockAccessStage }) };
  });
  window.history.replaceState({}, "", "/");
  mockWorkspace.mockImplementation(() => <p>Workspace loaded</p>);
});
afterEach(() => setLanguages({ interfaceLanguage: 'en', replyLanguage: 'auto' }, { remote: true }));

async function enterBirthDate(value) {
  const start=await screen.findByRole("button", { name: "Get started" });
  await waitFor(()=>expect(start).toBeEnabled());
  fireEvent.click(start);
  const field = screen.getByLabelText("Date of birth");
  fireEvent.change(field, { target: { value } });
  // Eligibility updates after the asynchronous server response.
  // eslint-disable-next-line testing-library/no-unnecessary-act
  await act(async () => { fireEvent.submit(field.form); });
}

test("blocks a younger visitor before loading the workspace, including after reload", async () => {
  const view = render(<Entry />);
  await enterBirthDate("2020-01-01");
  expect(screen.getByRole("heading", { name: "Classio is unavailable to you" })).toBeInTheDocument();
  expect(global.fetch).toHaveBeenCalledWith("/api/access", expect.objectContaining({ method: "POST" }));
  expect(mockWorkspace).not.toHaveBeenCalled();
  view.unmount();
  render(<Entry />);
  expect(await screen.findByRole("heading", { name: "Classio is unavailable to you" })).toBeInTheDocument();
});

test("loads the workspace only after the terms step for an eligible visitor", async () => {
  render(<Entry />);
  await enterBirthDate("2000-01-01");
  expect(mockWorkspace).not.toHaveBeenCalled();
  expect(mockAccessStage).toBe("eligible");
  fireEvent.click(screen.getByRole("checkbox"));
  // eslint-disable-next-line testing-library/no-unnecessary-act
  await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Open workspace" })); });
  expect(await screen.findByText("Workspace loaded")).toBeInTheDocument();
  expect(mockAccessStage).toBe("accepted");
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
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ provider: "auto", allowFallback: false, ...payload }),
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

test('older browsers without randomUUID still create IDs accepted by the real course schema',()=>{
  const {makeStudyId}=require('./studyTools'),{courseInput}=require('./shared/course-schema');
  for(const api of [{getRandomValues:bytes=>bytes.fill(171)},null]){
    const requestId=makeStudyId(api);
    expect(requestId).toMatch(/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/);
    expect(courseInput({tool:'create_course',requestId,arguments:{fileId:'source',topic:'Energy',difficulty:'beginner',lessons:2}})).not.toBeNull();
  }
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


test('returning Google sessions wait for account verification restore without flashing the age form (mocked auth)',async()=>{
  localStorage.setItem('classio_google_signin_requested','true');
  let complete;restoreSignedInAccount.mockReturnValue(new Promise(resolve=>{complete=resolve;}));
  render(<Entry/>);
  await waitFor(()=>expect(restoreSignedInAccount).toHaveBeenCalled());
  expect(screen.queryByLabelText('Date of birth')).not.toBeInTheDocument();
  expect(screen.getByRole('button',{name:'Get started'})).toBeDisabled();
  await act(async()=>{complete({uid:'test-account'});});
  expect(await screen.findByText('Workspace loaded')).toBeInTheDocument();
  expect(global.fetch.mock.calls.some(([,options])=>options?.body&&JSON.parse(options.body).action==='restore')).toBe(true);
});
test('a failed account restore preserves the retry state instead of marking a returning account unverified',async()=>{
  localStorage.setItem('classio_google_signin_requested','true');
  restoreSignedInAccount.mockRejectedValue(new Error('Temporary network problem'));
  render(<Entry/>);
  await screen.findByText('We couldn’t connect to Classio. Your saved verification has not been changed. We’ll try again when you’re online.');
  expect(screen.queryByRole('button',{name:'Get started'})).not.toBeInTheDocument();
});

test('the welcome page stays visible during a slow background check with no verification checking screen',async()=>{
  let complete;
  global.fetch=jest.fn(()=>new Promise(resolve=>{complete=resolve;}));
  render(<Entry/>);
  expect(screen.getByRole('heading',{name:'Your next great idea starts here.'})).toBeInTheDocument();
  expect(screen.queryByText(/Checking your saved verification/)).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Date of birth')).not.toBeInTheDocument();
  expect(mockWorkspace).not.toHaveBeenCalled();
  await act(async()=>complete({ok:true,json:async()=>({stage:'accepted'})}));
  expect(await screen.findByText('Workspace loaded')).toBeInTheDocument();
});
test('an HTML/static fallback cannot be mistaken for completed verification',async()=>{
  global.fetch=jest.fn(async()=>({ok:true,json:async()=>{throw new Error('HTML response');}}));
  render(<Entry/>);
  await screen.findByText('Classio’s access service could not be reached. Please try again shortly.');
  expect(mockWorkspace).not.toHaveBeenCalled();
});
test('a failed background check retries automatically when the connection returns',async()=>{
  global.fetch=jest.fn().mockRejectedValueOnce(new TypeError('offline')).mockResolvedValue({ok:true,json:async()=>({stage:'accepted'})});
  render(<Entry/>);await screen.findByRole('alert');
  await act(async()=>window.dispatchEvent(new Event('online')));
  expect(await screen.findByText('Workspace loaded')).toBeInTheDocument();
});
test('a browser that discards the eligibility cookie receives a real recovery message instead of access',async()=>{
  const normal=global.fetch;
  global.fetch=jest.fn(async(url,options)=>options.method==='POST'?{ok:true,json:async()=>({stage:'eligible'})}:normal(url,options));
  render(<Entry/>);await enterBirthDate('2000-01-01');
  await screen.findByText('Allow first-party cookies for Classio, then try again. Your browser did not keep the access decision.');
  expect(screen.getByLabelText('Date of birth')).toHaveValue('2000-01-01');
  expect(mockWorkspace).not.toHaveBeenCalled();
});
test('a stalled Firebase restoration is bounded and never marks the account as new',async()=>{
  jest.useFakeTimers();
  localStorage.setItem('classio_google_signin_requested','true');
  restoreSignedInAccount.mockReturnValue(new Promise(()=>{}));
  try{
    render(<Entry/>);
    await act(async()=>{await Promise.resolve();await Promise.resolve();});
    await act(async()=>jest.advanceTimersByTime(11001));
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.queryByLabelText('Date of birth')).not.toBeInTheDocument();
    expect(mockWorkspace).not.toHaveBeenCalled();
  }finally{jest.useRealTimers();}
});

test('a configuration error changes back to English without allowing unverified access', async () => {
  setLanguages({ interfaceLanguage: 'ar' }, { remote: true });
  global.fetch = jest.fn(async () => ({ ok: false, status: 503, headers: { get: () => null }, json: async () => ({ code: 'VERIFICATION_CONFIG_MISSING', error: 'Classio’s secure access service is not ready yet. Please try again shortly.' }) }));
  render(<Entry />);
  expect(await screen.findByRole('alert')).toHaveTextContent('كلاسيو غير متاح مؤقتًا. يرجى العودة لاحقًا.');
  fireEvent.change(screen.getByLabelText('لغة الواجهة'), { target: { value: 'en' } });
  expect(screen.getByRole('alert')).toHaveTextContent('Classio is temporarily unavailable. Please check back later.');
  expect(screen.getByRole('button', { name: 'Returning? Sign in with Google' })).toBeDisabled();
  expect(screen.queryByLabelText('Date of birth')).not.toBeInTheDocument();
  expect(mockWorkspace).not.toHaveBeenCalled();
});

test('a missing server secret does not start automatic retry polling and can recover after a manual check', async () => {
  jest.useFakeTimers();
  global.fetch = jest.fn(async () => ({ ok: false, status: 503, headers: { get: () => null }, json: async () => ({ code: 'VERIFICATION_CONFIG_MISSING' }) }));
  try {
    render(<Entry />);
    await act(async () => { await Promise.resolve(); await Promise.resolve(); });
    expect(screen.getByRole('alert')).toBeInTheDocument();
    await act(async () => jest.advanceTimersByTime(15000));
    expect(global.fetch).toHaveBeenCalledTimes(1);
    global.fetch.mockResolvedValue({ ok: true, json: async () => ({ stage: 'accepted' }) });
    fireEvent.click(screen.getByRole('button', { name: 'Check again' }));
    expect(await screen.findByText('Workspace loaded')).toBeInTheDocument();
  } finally { jest.useRealTimers(); }
});
