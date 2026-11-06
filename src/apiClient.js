// Portable timeout/abort composition, including browsers without AbortSignal.any.
export async function requestJson(path, { body, headers = {}, signal, timeout = 60000 } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort(signal?.reason);
  if (signal?.aborted) cancel();
  signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
  try {
    const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST',
      headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw Object.assign(new Error(data.error || 'This request could not finish. Please retry.'), { code: data.code, retryAfter: response.headers?.get('Retry-After') });
    return data;
  } catch (error) {
    if (timedOut) throw new Error('The request took too long. Your input is still here; please retry.');
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
