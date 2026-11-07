import { lazy, Suspense, useState } from "react";
import { checkBirthDate } from "./eligibility";
import { LegalLinks, LegalPage, legalPages } from "./Legal";
import "./entry.css";
import { readAccess, rememberAccess } from "./access";
import { ThemeSelector, useThemeMode } from "./theme";
import WorkspaceBoundary from "./WorkspaceBoundary";
import WorkspaceFrame from "./WorkspaceFrame";

const Workspace = lazy(() => import("./App"));

export default function Entry() {
  useThemeMode();
  const [access, setAccess] = useState(readAccess);
  const [screening, setScreening] = useState(false);
  const [error, setError] = useState("");

  const page = legalPages[window.location.pathname.replace(/\/$/, "")];
  if (page) return <LegalPage page={page} />;
  if (access === "accepted") {
    return <WorkspaceBoundary><Suspense fallback={<WorkspaceFrame />}><Workspace /></Suspense></WorkspaceBoundary>;
  }

  function setDecision(decision) {
    rememberAccess(decision);
    setAccess(decision);
  }

  function submitBirthDate(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const eligible = checkBirthDate(new FormData(form).get("birthDate"));
    if (eligible === null) {
      setError("Enter a valid date of birth.");
      return;
    }
    form.reset();
    setError("");
    setDecision(eligible ? "eligible" : "blocked");
  }

  return (
    <div className="entry-shell">
      <header className="entry-header"><a className="entry-brand" href="/"><img src="/ClassioLogo.png" width="32" height="32" alt="" />Classio</a><ThemeSelector /></header>
      <main className="entry-main">
        <div className="entry-intro">
          <p className="entry-kicker">Free study workspace · Ages 13+</p>
          <h1>Keep your study files, notes and flashcards together.</h1>
          <p>Organize class materials in folders, turn a reading into revision notes, and test what you remember with flashcards and quizzes.</p>
          <ul className="entry-feature-list">
            <li>Organize files by subject</li>
            <li>Make notes from your own materials</li>
            <li>Review with flashcards and quizzes</li>
          </ul>
          <p className="entry-session-note">Every study tool is free to use as a guest. Sign in with Google inside the workspace to save and sync progress. Guest work clears when you reload or close the session.</p>
        </div>
        <section className="entry-card" aria-labelledby="entry-step">
          {access === "blocked" ? (
            <>
              <h2 id="entry-step">Classio is unavailable to you</h2>
              <p>Based on the date you entered, you cannot use the study workspace. You can still read our privacy policy.</p>
            </>
          ) : access === "eligible" ? (
            <form onSubmit={event => { event.preventDefault(); setDecision("accepted"); }}>
              <h2 id="entry-step">Review the terms</h2>
              <label className="entry-agreement">
                <input type="checkbox" required />
                <span>I agree to the <a href="/terms" target="_blank" rel="noreferrer">Terms and Conditions</a> and acknowledge the <a href="/privacy" target="_blank" rel="noreferrer">Privacy Policy</a>.</span>
              </label>
              <button className="entry-button">Open workspace</button>
            </form>
          ) : screening ? (
            <form onSubmit={submitBirthDate}>
              <h2 id="entry-step">Check eligibility</h2>
              <p>Your date of birth stays in this form. Classio remembers the eligibility result in this browser.</p>
              <label htmlFor="birth-date">Date of birth</label>
              <input id="birth-date" name="birthDate" type="date" required onInput={() => setError("")} />
              {error && <p className="entry-error" role="alert">{error}</p>}
              <button className="entry-button">Continue</button>
            </form>
          ) : (
            <>
              <h2 id="entry-step">Open Classio</h2>
              <p>Check eligibility to open the workspace. You can sign in later if you want to save your work.</p>
              <button className="entry-button" onClick={() => setScreening(true)}>Get started</button>
            </>
          )}
        </section>
      </main>
      <LegalLinks />
    </div>
  );
}
