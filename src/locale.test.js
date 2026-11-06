import { act, fireEvent, render, screen } from '@testing-library/react';
import { useMemo } from 'react';
import LanguageSettings from './LanguageSettings';
import { getLanguagePreferences, getReplyLanguage, setLanguages, translate, useLocale } from './locale';
import { sendStudyRequest } from './studyTools';
jest.mock('./firebase', () => ({ accountHeaders:jest.fn(async () => ({})) }));
let originalFetch;
beforeEach(() => { originalFetch=global.fetch; setLanguages({interfaceLanguage:'en',replyLanguage:'auto'},{remote:true}); });
afterEach(() => { global.fetch=originalFetch; act(()=>setLanguages({interfaceLanguage:'en',replyLanguage:'auto'},{remote:true})); jest.restoreAllMocks(); });

test('language controls translate immediately, set RTL and persist independent choices', () => {
  render(<LanguageSettings />);
  fireEvent.change(screen.getByLabelText('AI response language'),{target:{value:'French'}});
  fireEvent.change(screen.getByLabelText('Interface language'),{target:{value:'ar'}});
  expect(document.documentElement).toHaveAttribute('lang','ar');
  expect(document.documentElement).toHaveAttribute('dir','rtl');
  expect(screen.getByLabelText('لغة إجابات الذكاء الاصطناعي')).toHaveValue('French');
  expect(translate('Generate AI notes')).toBe('إنشاء ملاحظات ذكية');
  expect(localStorage.getItem('classio_interface_language')).toBe('ar');
  expect(localStorage.getItem('classio_lang')).toBe('French');
  fireEvent.change(screen.getByLabelText('لغة الواجهة'),{target:{value:'en'}});
  expect(document.documentElement).toHaveAttribute('dir','ltr');
  expect(screen.getByLabelText('AI response language')).toHaveValue('French');
  expect(screen.getByText('Choose the language of navigation, settings and study controls.')).toBeInTheDocument();
});

test('a remote preference update is preserved when the next local setting changes', () => {
  render(<LanguageSettings />);
  act(()=>setLanguages({replyLanguage:'Arabic'},{remote:true}));
  fireEvent.change(screen.getByLabelText('Interface language'),{target:{value:'ar'}});
  expect(getLanguagePreferences()).toEqual({interfaceLanguage:'ar',replyLanguage:'Arabic'});
});

test('new AI requests use the current preference without a refresh and never send client-only language fields', async () => {
  global.fetch=jest.fn(async ()=>({ok:true,json:async()=>({text:'A real response would come from the provider.',provider:'test-fixture'})}));
  const payload={messages:[{role:'system',content:'Explain carefully.'},{role:'user',content:'Explain energy.'}]};
  setLanguages({replyLanguage:'Arabic'});
  await sendStudyRequest(payload);
  const first=JSON.parse(global.fetch.mock.calls[0][1].body);
  expect(first.messages[0].content).toContain('Respond in Arabic');
  expect(payload.messages[0].content).toBe('Explain carefully.');
  setLanguages({replyLanguage:'French'});
  await sendStudyRequest(payload);
  expect(JSON.parse(global.fetch.mock.calls[1][1].body).messages[0].content).toContain('Respond in French');
  await sendStudyRequest({...payload,replyLanguage:'auto'});
  const last=JSON.parse(global.fetch.mock.calls[2][1].body);
  expect(last.messages[0].content).toBe('Explain carefully.');
  expect(last).not.toHaveProperty('replyLanguage');
});

test('blocked browser storage still allows an immediate in-memory language preference', () => {
  jest.spyOn(Storage.prototype,'setItem').mockImplementation(()=>{throw new Error('Storage blocked');});
  setLanguages({interfaceLanguage:'ar',replyLanguage:'Arabic'});
  expect(getReplyLanguage()).toBe('Arabic');
  expect(document.documentElement.dir).toBe('rtl');
});

test('memoized labels and stored message keys follow an Arabic to English round trip', () => {
  const message = { key: 'Saved version: {name}.', values: { name: 'مراجعتي — Chapter 1' } };
  function Preview() {
    const { t } = useLocale();
    const label = useMemo(() => t('Study notes'), [t]);
    return <><label>{label}<textarea aria-label={label} defaultValue="كتابتي الأصلية" /></label><p role="status">{t(message)}</p></>;
  }
  render(<Preview />);
  act(() => setLanguages({ interfaceLanguage: 'ar' }));
  expect(screen.getByRole('status')).toHaveTextContent('حُفظت النسخة: مراجعتي — Chapter 1.');
  act(() => setLanguages({ interfaceLanguage: 'en' }));
  expect(screen.getByRole('textbox', { name: 'Study notes' })).toHaveValue('كتابتي الأصلية');
  expect(screen.getByRole('status')).toHaveTextContent('Saved version: مراجعتي — Chapter 1.');
});
