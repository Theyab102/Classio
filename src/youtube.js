const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
const HOSTS = new Set(['youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com', 'youtu.be', 'www.youtu.be']);

export function youtubeVideo(input) {
  let url;
  try { url = new URL(input); } catch { return null; }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.port || !HOSTS.has(url.hostname)) return null;
  const parts = url.pathname.split('/').filter(Boolean);
  const id = url.hostname.endsWith('youtu.be') ? (parts.length === 1 ? parts[0] : null)
    : url.pathname === '/watch' ? url.searchParams.get('v')
      : ['shorts', 'live', 'embed'].includes(parts[0]) && parts.length === 2 ? parts[1] : null;
  if (!VIDEO_ID.test(id || '')) return null;
  return { id, url: `https://www.youtube.com/watch?v=${id}`, thumbnail: `https://i.ytimg.com/vi/${id}/hqdefault.jpg` };
}

export function normalizeStudyLink(input) {
  const url = new URL(input.trim());
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Enter a public http or https link without embedded credentials.');
  if (HOSTS.has(url.hostname)) {
    const video = youtubeVideo(url.href);
    if (!video) throw new Error('Enter a valid YouTube watch, Shorts, live or youtu.be video link.');
    return video.url;
  }
  if (url.hostname.endsWith('.youtube.com') || url.hostname.endsWith('.youtu.be')) throw new Error('This YouTube URL format is not supported.');
  return url.href;
}
