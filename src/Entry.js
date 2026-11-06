import { lazy, Suspense, useEffect, useState } from 'react';
import { LegalLinks, LegalPage, legalPages } from './Legal';
import { requestJson } from './apiClient';
import './entry.css';

const Workspace = lazy(() => import('./App'));

async function accessRequest(payload, headers = {}) {
  const data = await requestJson('/api/access', { body: payload, headers, timeout: 12000 });
  return data.stage;
}

export default function Entry() {
  const [access, setAccess] = useState('loading');
  const [screening, setScreening] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const page = legalPages[window.location.pathname.replace(/\/$/, '')];

  useEffect(() => {
    if (page) return;
    let active = true;
    accessRequest().then(async stage => {
      let requested=false;try{requested=localStorage.getItem('classio_google_signin_requested')==='true';}catch{}
      if(stage==='new'&&requested){
        const firebase=await import('./firebase');
        const user=await firebase.restoreSignedInAccount();
        if(user)stage=await accessRequest({action:'restore'},await firebase.accountHeaders());
      }
      if(active)setAccess(stage);
    }).catch(() => {
      if (active) { setAccess('unavailable'); setError('Could not check your saved verification. Check your connection and retry.'); }
    });
    return () => { active = false; };
  }, [page]);

  async function run(payload) {
    if (pending) return;
    setPending(true); setError('');
    try { setAccess(await accessRequest(payload)); }
    catch (failure) { setError(failure.message); }
    finally { setPending(false); }
  }

  async function restoreAccount() {
    if (pending) return;
    setPending(true); setError('');
    try {
      const firebase = await import('./firebase');
      await firebase.signInGoogle();
      await firebase.authReady.catch(() => {});
      if (firebase.auth.currentUser) setAccess(await accessRequest({ action: 'restore' }, await firebase.accountHeaders()));
    } catch (failure) {
      const { authError } = await import('./firebase');
      setError(failure.code ? authError(failure) : failure.message);
    } finally { setPending(false); }
  }

  if (page) return <LegalPage page={page} />;
  if (access === 'loading') return <p className="entry-loading" role="status">Checking your saved verification…</p>;
  if (access === 'accepted') return <Suspense fallback={<p className="entry-loading" role="status">Loading Classio…</p>}><Workspace /></Suspense>;

  return <div className="entry-shell">
    <header className="entry-header"><a className="entry-brand" href="/"><img src="/ClassioLogo.png" width="32" height="32" alt="" />Classio</a></header>
    <main className="entry-main">
      <div className="entry-intro">
        <p className="entry-kicker">A little clarity. A lot of possibility.</p>
        <h1>Your next great idea starts here.</h1>
        <p>A thoughtful home for your study files, notes and flashcards. Learn at your own pace, with a little help when you need it.</p>
        <ul className="entry-feature-list"><li>One workspace for every subject</li><li>Your materials, made easier to understand</li><li>Free to explore. Yours to make progress.</li></ul>
      </div>
      <section className="entry-card" aria-labelledby="entry-step" aria-busy={pending}>
        <p className="entry-kicker">Welcome to Classio</p>
        {access === 'blocked' ? <><h2 id="entry-step">Classio is unavailable to you</h2><p>Based on the date you entered, you cannot use the study workspace. You can still read our privacy policy.</p></> : access === 'eligible' ?
          <form onSubmit={event => { event.preventDefault(); run({ action: 'accept', termsAccepted: true }); }}>
            <h2 id="entry-step">Make yourself at home</h2><p>Take a moment to review how Classio works.</p>
            <label className="entry-agreement"><input type="checkbox" required disabled={pending} /><span>I agree to the <a href="/terms" target="_blank" rel="noreferrer">Terms and Conditions</a> and acknowledge the <a href="/privacy" target="_blank" rel="noreferrer">Privacy Policy</a>.</span></label>
            <button className="entry-button" disabled={pending}>{pending ? 'Opening…' : 'Open workspace'}</button>
          </form> : screening ? <form onSubmit={event => { event.preventDefault(); const form = event.currentTarget; const birthDate = new FormData(form).get('birthDate'); run({ action: 'screen', birthDate }); }}>
            <h2 id="entry-step">A quick eligibility check</h2><p>Classio is for people aged 13 and over. We keep the verification result, not your date of birth.</p>
            <label htmlFor="birth-date">Date of birth</label><input id="birth-date" name="birthDate" type="date" required disabled={pending} aria-describedby={error ? 'access-error' : undefined} />
            <button className="entry-button" disabled={pending}>{pending ? 'Checking…' : 'Continue'}</button>
          </form> : <>
            <h2 id="entry-step">Your space to learn</h2><p>Check eligibility once to get started. This browser remembers your completed verification.</p>
            {access === 'unavailable' ? <button className="entry-button" onClick={async () => { setAccess('loading'); try { setAccess(await accessRequest()); setError(''); } catch { setAccess('unavailable'); } }}>Retry verification check</button> : <button className="entry-button" onClick={() => setScreening(true)}>Get started</button>}
            <button className="entry-button secondary" onClick={restoreAccount} disabled={pending}>{pending ? 'Connecting…' : 'Returning? Sign in with Google'}</button>
          </>}
        {error && <p id="access-error" className="entry-error" role="alert">{error}</p>}
      </section>
    </main><LegalLinks />
  </div>;
}
