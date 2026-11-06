import { useEffect, useState } from 'react';
import { useLocale } from './locale';

export default function AISettings() {
  const { t } = useLocale();
  const [preferences, setPreferences] = useState(() => {
    try { return JSON.parse(localStorage.getItem('classio_ai_preferences')) || { provider: 'auto', allowFallback: false }; }
    catch { return { provider: 'auto', allowFallback: false }; }
  });
  const [available, setAvailable] = useState(null);
  useEffect(() => { let active = true; fetch('/api/study').then(response => response.json()).then(data => { if (active) setAvailable(data.providers); }).catch(() => {}); return () => { active = false; }; }, []);
  function update(next) {
    setPreferences(next);
    try { localStorage.setItem('classio_ai_preferences', JSON.stringify(next)); } catch {}
  }
  return <fieldset className="ai-preferences"><legend>{t("AI provider")}</legend>
    <label>{t("Use")}<select value={preferences.provider} onChange={event => update({ ...preferences, provider: event.target.value })}>
      <option value="auto">{t("Auto (Groq first)")}</option>
      <option value="groq">Groq{available && !available.groq ? ' · '+t('not configured') : ''}</option>
      <option value="gemini">Gemini{available && !available.gemini ? ' · '+t('not enabled') : ''}</option>
    </select></label>
    <label className="checkbox-label"><input type="checkbox" checked={preferences.allowFallback === true} onChange={event => update({ ...preferences, allowFallback: event.target.checked })} />{t("Allow one retry with the other enabled provider if the first is busy.")}</label>
    <p>{t("Prompts and selected study content go to the provider you choose. Enabling fallback permits sharing that request with either provider. Auto uses a configured provider; it does not retry another provider without this permission.")}</p>
  </fieldset>;
}
