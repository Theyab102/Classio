class ProviderError extends Error {
  constructor(status, code, message, provider) {
    super(message);
    Object.assign(this, { status, code, provider });
  }
}

function redactCredentials(text) {
  for (const name of ['GROQ_API_KEY', 'GEMINI_API_KEY', 'CLASSIO_TTS_SIGNING_SECRET', 'AI_RATE_LIMIT_SECRET', 'CLASSIO_SESSION_SECRET', 'FIREBASE_SERVICE_ACCOUNT_JSON']) {
    const value = process.env[name];
    if (value?.length >= 8) text = text.split(value).join('[redacted credential]');
  }
  return text;
}

function providers() {
  return {
    groq: Boolean(process.env.GROQ_API_KEY),
    // Operator opt-in requires checking current pricing and audience/data terms.
    gemini: Boolean(process.env.GEMINI_API_KEY && process.env.GEMINI_MODEL &&
      process.env.GEMINI_ENABLED === 'true' && process.env.GEMINI_FREE_TIER_VERIFIED === 'true'),
  };
}

async function checkedFetch(provider, url, options) {
  let response;
  try { response = await fetch(url, { ...options, signal: AbortSignal.timeout(25_000) }); }
  catch { throw new ProviderError(503, 'PROVIDER_TIMEOUT', 'The AI service did not respond in time. Please retry.', provider); }
  if (!response.ok) {
    const status = response.status;
    // Never forward a provider error body, headers, credentials or stack trace.
    const [code, message] = status === 429
      ? ['PROVIDER_RATE_LIMIT', 'This AI provider has reached its usage limit. Wait and retry, or choose another configured provider.']
      : status === 401 || status === 403
        ? ['PROVIDER_AUTH_FAILED', 'The AI provider rejected the server configuration. The operator needs to check its key and permissions.']
        : status === 400 || status === 404
          ? ['PROVIDER_MODEL_UNAVAILABLE', 'The selected AI model or request is not supported. The operator needs to check the model setting.']
          : ['PROVIDER_UNAVAILABLE', 'The AI provider is temporarily unavailable. Please retry.'];
    throw new ProviderError(status === 429 ? 429 : status >= 500 ? 503 : 502, code, message, provider);
  }
  return response;
}

function geminiContent(messages) {
  const instruction = messages.filter(message => message.role === 'system').map(message => message.content).join('\n\n');
  const contents = messages.filter(message => message.role !== 'system').map(message => ({
    role: message.role === 'assistant' ? 'model' : 'user',
    parts: typeof message.content === 'string' ? [{ text: message.content }] : message.content.map(part => {
      if (part.type === 'text') return { text: part.text };
      const [, mimeType, data] = /^data:([^;]+);base64,(.+)$/.exec(part.image_url.url);
      return { inlineData: { mimeType, data } };
    }),
  }));
  return { ...(instruction ? { systemInstruction: { parts: [{ text: instruction }] } } : {}), contents };
}

async function generateWith(provider, input) {
  if (provider === 'gemini') {
    const model = process.env.GEMINI_MODEL;
    if (!/^[a-zA-Z0-9.-]{1,100}$/.test(model || '')) throw new ProviderError(503, 'MODEL_NOT_CONFIGURED', 'The Gemini model setting is invalid.', provider);
    const response = await checkedFetch(provider, `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'x-goog-api-key': process.env.GEMINI_API_KEY },
      body: JSON.stringify({ ...geminiContent(input.messages), generationConfig: { maxOutputTokens: input.maxTokens } }),
    });
    const data = await response.json();
    return (data.candidates?.[0]?.content?.parts || []).filter(part => !part.thought).map(part => part.text || '').join('');
  }
  const model = input.vision ? process.env.GROQ_VISION_MODEL : (process.env.GROQ_TEXT_MODEL || 'openai/gpt-oss-120b');
  if (!model) throw new ProviderError(503, 'VISION_NOT_CONFIGURED', 'Image analysis needs a supported vision model configured on the server.', provider);
  const response = await checkedFetch(provider, 'https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
    body: JSON.stringify({ model, messages: input.messages, max_completion_tokens: input.maxTokens }),
  });
  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

async function generate(input, selection = {}, eligibility = {}) {
  const available = providers();
  if(eligibility.adult!==true && available.gemini) {
    available.gemini=false;
    if(selection.provider==='gemini' || ((selection.provider||'auto')==='auto'&&!available.groq))throw new ProviderError(403,'PROVIDER_AGE_RESTRICTED','Gemini is unavailable for this session under the deployment’s audience rules. Choose Groq.','gemini');
  }
  const selected = selection.provider || 'auto';
  if (!['auto', 'groq', 'gemini'].includes(selected)) throw new ProviderError(400, 'INVALID_PROVIDER', 'Choose Auto, Groq or Gemini.');
  if (selected === 'auto' && !available.groq && !available.gemini) throw new ProviderError(503, 'PROVIDER_NOT_CONFIGURED', 'No AI provider is enabled on this deployment. The operator must configure Groq or an eligible Gemini project and redeploy.');
  const first = selected === 'auto' ? (available.groq ? 'groq' : 'gemini') : selected;
  if (!available[first]) throw new ProviderError(503, 'PROVIDER_NOT_CONFIGURED', `${first === 'gemini' ? 'Gemini' : 'Groq'} is not enabled on this deployment. Choose another provider or contact the operator.`, first);
  try {
    const text = await generateWith(first, input);
    if (typeof text !== 'string' || !text.trim()) throw new ProviderError(502, 'EMPTY_RESPONSE', 'The AI returned no usable content. Please retry.', first);
    return { text: redactCredentials(text), provider: first, fallback: false };
  } catch (error) {
    const second = first === 'groq' ? 'gemini' : 'groq';
    if (selection.allowFallback !== true || !available[second] || !['PROVIDER_TIMEOUT', 'PROVIDER_RATE_LIMIT', 'PROVIDER_UNAVAILABLE'].includes(error.code)) throw error;
    const text = await generateWith(second, input);
    if (typeof text !== 'string' || !text.trim()) throw new ProviderError(502, 'EMPTY_RESPONSE', 'The alternate provider returned no usable content. Please retry.', second);
    return { text: redactCredentials(text), provider: second, fallback: true };
  }
}

async function transcribe(input) {
  if (!process.env.GROQ_API_KEY) throw new ProviderError(503, 'PROVIDER_NOT_CONFIGURED', 'Audio transcription requires Groq to be configured.', 'groq');
  const form = new FormData();
  form.set('model', 'whisper-large-v3-turbo');
  form.set('file', new Blob([Buffer.from(input.audio, 'base64')]), input.filename);
  const response = await checkedFetch('groq', 'https://api.groq.com/openai/v1/audio/transcriptions', {
    method: 'POST', headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` }, body: form,
  });
  const data = await response.json();
  if (typeof data.text !== 'string' || !data.text.trim()) throw new ProviderError(502, 'EMPTY_RESPONSE', 'No speech was transcribed. Please retry.', 'groq');
  return { text: redactCredentials(data.text), provider: 'groq', fallback: false };
}
module.exports = { generate, transcribe, providers, ProviderError, geminiContent };
