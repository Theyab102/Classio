import { normalizeStudyLink, youtubeVideo } from './youtube';

const id = 'dQw4w9WgXcQ';
test.each([
  `https://www.youtube.com/watch?v=${id}&list=test&t=15`,
  `https://youtu.be/${id}?si=extra`,
  `https://m.youtube.com/shorts/${id}`,
  `https://www.youtube.com/live/${id}?feature=share`,
  `https://youtube.com/embed/${id}`,
])('normalizes a supported YouTube video without trusting its raw embed URL: %s', url => {
  expect(youtubeVideo(url)?.id).toBe(id);
  expect(normalizeStudyLink(url)).toBe(`https://www.youtube.com/watch?v=${id}`);
});
test.each(['https://youtube.com.evil.test/watch?v=' + id, 'https://evil.test/?v=' + id, 'javascript:alert(1)', 'https://user:pass@youtube.com/watch?v=' + id, 'https://youtube.com/watch?v=short', 'https://youtu.be/' + id + '/extra'])('rejects unsafe or invalid video URLs: %s', url => {
  expect(youtubeVideo(url)).toBeNull();
});
test('refuses unsafe study link schemes and malformed YouTube IDs', () => {
  expect(() => normalizeStudyLink('javascript:alert(1)')).toThrow();
  expect(() => normalizeStudyLink('https://youtube.com/watch?v=invalid')).toThrow();
  expect(normalizeStudyLink('https://en.wikipedia.org/wiki/Force')).toBe('https://en.wikipedia.org/wiki/Force');
});
