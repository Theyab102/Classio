import { requestJson } from './apiClient';
import { setLanguages, translate } from './locale';
let original;
beforeEach(() => { original = global.fetch; });
afterEach(() => { global.fetch = original; jest.useRealTimers(); setLanguages({ interfaceLanguage: 'en' }, { remote: true }); });

test('timeout rejects with a recoverable message and aborts the actual fetch', async () => {
  jest.useFakeTimers();
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))));
  const request = requestJson('/api/study', { body: { question: 'Test question' }, timeout: 1000 });
  jest.advanceTimersByTime(1001);
  await expect(request).rejects.toThrow('Your input is still here');
  expect(global.fetch.mock.calls[0][1].signal.aborted).toBe(true);
});

test('leaving a screen cancels the network request without replacing it with a timeout error', async () => {
  const controller = new AbortController();
  global.fetch = jest.fn((_url, options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))));
  const request = requestJson('/api/study', { body: {}, signal: controller.signal });
  controller.abort();
  await expect(request).rejects.toMatchObject({ name: 'AbortError' });
});

test('HTTP quota errors retain a safe code and retry delay', async () => {
  global.fetch = jest.fn(async () => ({ ok: false, headers: { get: () => '60' }, json: async () => ({ code: 'USAGE_LIMIT', error: 'Try again later.' }) }));
  await expect(requestJson('/api/study', { body: {} })).rejects.toMatchObject({ code: 'USAGE_LIMIT', retryAfter: '60', message: 'Try again later.' });
});

test('an API error created in Arabic remains translatable after returning to English', async () => {
  setLanguages({ interfaceLanguage: 'ar' }, { remote: true });
  global.fetch = jest.fn(async () => ({ ok: false, status: 503, headers: { get: () => null }, json: async () => ({ code: 'AI_TIMEOUT', error: 'The AI service did not respond in time. Please retry.' }) }));
  const error = await requestJson('/api/study', { body: {} }).catch(failure => failure);
  expect(error).toMatchObject({ code: 'AI_TIMEOUT', status: 503, message: 'The AI service did not respond in time. Please retry.' });
  expect(translate(error.message)).toBe('لم تستجب خدمة الذكاء الاصطناعي في الوقت المحدد. حاول مجددًا.');
  setLanguages({ interfaceLanguage: 'en' }, { remote: true });
  expect(translate(error.message)).toBe('The AI service did not respond in time. Please retry.');
});
