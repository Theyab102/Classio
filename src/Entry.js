import { lazy, Suspense, useState } from 'react';
import { LegalLinks, LegalPage, legalPages } from './Legal';
import useAccessSession from './useAccessSession';
import { setLanguages, useLocale } from './locale';
import './entry.css';

const Workspace = lazy(() => import('./App'));

function EntryShell({children}) {
  const {t,locale}=useLocale();
  return <div className="entry-shell">
    <header className="entry-header"><a className="entry-brand" href="/"><img src="/ClassioLogo.png" width="32" height="32" alt=""/>Classio</a><label className="entry-language"><span className="sr-only">{t('Interface language')}</span><select aria-label={t('Interface language')} value={locale} onChange={event=>setLanguages({interfaceLanguage:event.target.value})}><option value="en">English</option><option value="ar">العربية</option></select></label></header>
    <main className="entry-main"><div className="entry-intro"><p className="entry-kicker">{t('A little clarity. A lot of possibility.')}</p><h1>{t('Your next great idea starts here.')}</h1><p>{t('A thoughtful home for your study files, notes and flashcards. Learn at your own pace, with a little help when you need it.')}</p><ul className="entry-feature-list"><li>{t('One workspace for every subject')}</li><li>{t('Your materials, made easier to understand')}</li><li>{t('Free to explore. Yours to make progress.')}</li></ul></div>{children}</main><LegalLinks/>
  </div>;
}
function OpeningWorkspace(){
  const {t}=useLocale();
  return <EntryShell><section className="entry-card entry-workspace-preview" aria-label={t('Your workspace')} aria-busy="true"><p className="entry-kicker">Classio</p><h2>{t('Your space to learn')}</h2><div className="entry-skeleton" aria-hidden="true"><i/><i/><i/></div></section></EntryShell>;
}
export default function Entry() {
  const {t}=useLocale();
  const page=legalPages[window.location.pathname.replace(/\/$/,'')];
  const {stage,checking,pending,error,configurationMissing,check,run}=useAccessSession(!page);
  const [screening,setScreening]=useState(false);
  async function restoreAccount(){
    await run({action:'restore'},async()=>{
      const firebase=await import('./firebase');
      try{await firebase.signInGoogle();return await firebase.accountHeaders();}
      catch(failure){throw Object.assign(new Error(firebase.authError(failure)),{code:failure.code||'auth/failed'});}
    });
  }
  if(page)return <LegalPage page={page}/>;
  if(stage==='accepted')return <Suspense fallback={<OpeningWorkspace/>}><Workspace/></Suspense>;
  const waiting=checking||pending||stage==='checking';
  return <EntryShell><section className="entry-card" aria-labelledby="entry-step" aria-busy={waiting}>
    <p className="entry-kicker">{t('Welcome to Classio')}</p>
    {stage==='blocked'?<><h2 id="entry-step">{t('Classio is unavailable to you')}</h2><p>{t('Based on the date you entered, you cannot use the study workspace. You can still read our privacy policy.')}</p></>:stage==='eligible'?<form onSubmit={event=>{event.preventDefault();run({action:'accept',termsAccepted:true});}}>
      <h2 id="entry-step">{t('Make yourself at home')}</h2><p>{t('Take a moment to review how Classio works.')}</p><label className="entry-agreement"><input type="checkbox" required disabled={waiting}/><span>{t('I agree to the ')}<a href="/terms" target="_blank" rel="noreferrer">{t('Terms and Conditions')}</a>{t(' and acknowledge the ')}<a href="/privacy" target="_blank" rel="noreferrer">{t('Privacy Policy')}</a>.</span></label><button className="entry-button" disabled={waiting}>{t(pending?'Opening…':'Open workspace')}</button>
    </form>:screening&&stage==='new'?<form onSubmit={event=>{event.preventDefault();run({action:'screen',birthDate:new FormData(event.currentTarget).get('birthDate')});}}>
      <h2 id="entry-step">{t('A quick eligibility check')}</h2><p>{t('Classio is for people aged 13 and over. We keep the verification result, not your date of birth.')}</p><label htmlFor="birth-date">{t('Date of birth')}</label><input id="birth-date" name="birthDate" type="date" required disabled={waiting} aria-describedby={error?'access-error':undefined}/><button className="entry-button" disabled={waiting}>{t(pending?'Continuing…':'Continue')}</button>
    </form>:<>
      <h2 id="entry-step">{t('Your space to learn')}</h2><p>{t('Bring your materials. Find a little clarity, one idea at a time.')}</p>
      {stage==='unavailable'?<button className="entry-button" onClick={check} disabled={waiting}>{t(configurationMissing?'Check again':'Try again')}</button>:<button className="entry-button" onClick={()=>setScreening(true)} disabled={waiting}>{t('Get started')}</button>}
      <button className="entry-button secondary" onClick={restoreAccount} disabled={waiting||configurationMissing}>{t(pending?'Connecting…':'Returning? Sign in with Google')}</button>
    </>}
    {error&&<p id="access-error" className="entry-error" role="alert">{t(error)}</p>}
  </section></EntryShell>;
}
