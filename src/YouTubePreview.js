import { useEffect, useState } from 'react';
import { youtubeVideo } from './youtube';

export default function YouTubePreview({ url }) {
  const video = youtubeVideo(url);
  const id = video?.id;
  const [title, setTitle] = useState('YouTube video');
  const [notice, setNotice] = useState('');
  const [enabled, setEnabled] = useState(false);
  useEffect(() => {
    setEnabled(false); setTitle('YouTube video'); setNotice('');
    if (!id) return;
    const controller = new AbortController();
    fetch(`https://www.youtube.com/oembed?url=${encodeURIComponent(`https://www.youtube.com/watch?v=${id}`)}&format=json`, { signal: controller.signal })
      .then(async response => { if (!response.ok) throw new Error(); return response.json(); })
      .then(data => setTitle(data.title || 'YouTube video')).catch(() => { if (!controller.signal.aborted) setNotice('Video details could not be loaded. The video may be unavailable or the connection blocked.'); });
    const timeout = setTimeout(() => controller.abort(), 10000);
    return () => { clearTimeout(timeout); controller.abort(); };
  }, [id]);
  if (!video) return null;
  return <section className="youtube-preview" aria-label="Saved YouTube video"><h3>{title}</h3>
    {enabled ? <iframe key={id} title={title} src={`https://www.youtube-nocookie.com/embed/${id}`} allow="encrypted-media; picture-in-picture" allowFullScreen referrerPolicy="strict-origin-when-cross-origin" />
      : <button type="button" onClick={() => setEnabled(true)}>Load video from YouTube</button>}
    {notice && <p role="status">{notice}</p>}<p>If the video is private, restricted or cannot be embedded, <a href={video.url} target="_blank" rel="noreferrer">open it on YouTube</a>.</p>
  </section>;
}
