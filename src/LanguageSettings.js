import { getLanguagePreferences, interfaceLanguages, replyLanguages, setLanguages, useLocale } from './locale';

const nativeNames = { auto:'Auto detect', English:'English', Arabic:'العربية', French:'Français', Spanish:'Español', German:'Deutsch', Italian:'Italiano', Portuguese:'Português', Turkish:'Türkçe', Hindi:'हिन्दी', Urdu:'اردو', Chinese:'中文', Japanese:'日本語', Korean:'한국어', Russian:'Русский' };
export default function LanguageSettings() {
  const { t } = useLocale();
  function update(next) { setLanguages({ ...getLanguagePreferences(), ...next }); }
  return <div className="language-settings">
    <label>{t('Interface language')}<select aria-label={t('Interface language')} value={getLanguagePreferences().interfaceLanguage} onChange={event => update({ interfaceLanguage:event.target.value })}>{interfaceLanguages.map(item => <option key={item.value} value={item.value}>{t(item.label)}</option>)}</select><span>{t('Choose the language of navigation, settings and study controls.')}</span></label>
    <label>{t('AI response language')}<select aria-label={t('AI response language')} value={getLanguagePreferences().replyLanguage} onChange={event => update({ replyLanguage:event.target.value })}>{replyLanguages.map(value => <option key={value} value={value}>{t(nativeNames[value])}</option>)}</select><span>{t('Choose the language for new AI responses. Existing notes and uploaded files are kept as they are.')}</span></label>
    <p>{t('Arabic includes right-to-left navigation.')}</p>
  </div>;
}
