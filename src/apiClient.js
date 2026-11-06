// Portable timeout/abort composition, including browsers without AbortSignal.any.
export async function requestJson(path, { body, headers = {}, signal, timeout = 60000 } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort(signal?.reason);
  if (signal?.aborted) cancel();
  signal?.addEventListener('abort', cancel, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
  try {
    const response = await fetch(path, { method: body === undefined ? 'GET' : 'POST', credentials:'same-origin', cache:'no-store',
      headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...headers },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    // Keep canonical text in async state. Views translate it using the current
    // interface language, including when that language changes after failure.
    if (!response.ok) throw Object.assign(new Error(data.error || (typeof data.detail==='string'?data.detail:'This request could not finish. Please retry.')), { status:response.status, code: data.code, retryAfter: response.headers?.get('Retry-After') });
    return data;
  } catch (error) {
    if (timedOut) throw new Error('The request took too long. Your input is still here; please retry.');
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}

// Plain text counterpart with portable cancellation for older iOS browsers.
export async function requestText(path,{signal,timeout=12000,...options}={}){
  const controller=new AbortController();const cancel=()=>controller.abort();
  if(signal?.aborted)cancel();signal?.addEventListener('abort',cancel,{once:true});
  const timer=setTimeout(cancel,timeout);
  try{const response=await fetch(path,{...options,credentials:'omit',referrerPolicy:'no-referrer',signal:controller.signal});if(!response.ok)throw new Error('This source is unavailable.');return await response.text();}
  finally{clearTimeout(timer);signal?.removeEventListener('abort',cancel);}
}
