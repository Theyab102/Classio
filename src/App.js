import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import ReactDOM from "react-dom";
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut, onAuthStateChanged } from "firebase/auth";
import { getFirestore, doc, setDoc, getDoc, collection, addDoc, onSnapshot, updateDoc, deleteDoc, serverTimestamp, query, orderBy, limit, getDocs, arrayUnion } from "firebase/firestore";
import { getStorage, ref as storageRef, uploadBytes, getDownloadURL } from "firebase/storage";
import { getDatabase, ref as dbRef, set as dbSet, get as dbGet } from "firebase/database";

// ─── FIREBASE ─────────────────────────────────────────────────────────────────
const firebaseConfig = {
  apiKey: "AIzaSyCDDXRAxLwNmzSYxs0HnnjV2TfhbqSdiag",
  authDomain: "classio-4378f.firebaseapp.com",
  projectId: "classio-4378f",
  storageBucket: "classio-4378f.firebasestorage.app",
  messagingSenderId: "595968221954",
  appId: "1:595968221954:web:cdefee80f05999f8bf181b",
  databaseURL: "https://classio-4378f-default-rtdb.firebaseio.com",
};
const firebaseApp = initializeApp(firebaseConfig);
const auth = getAuth(firebaseApp);
const db = getFirestore(firebaseApp);
const storage = getStorage(firebaseApp);
const rtdb = getDatabase(firebaseApp);
const googleProvider = new GoogleAuthProvider();

// ─── GROQ AI (FREE) ──────────────────────────────────────────────────────────
const GROQ_KEY = process.env.REACT_APP_GROQ_KEY || "";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const GEMINI_KEYS = [process.env.REACT_APP_GEMINI_KEY_1||"",process.env.REACT_APP_GEMINI_KEY_2||"",process.env.REACT_APP_GEMINI_KEY_3||""].filter(Boolean);

// ─── AI BACKEND — MULTI-KEY ROTATOR + FAILOVER ───────────────────────────────
const OPENROUTER_KEY = process.env.REACT_APP_OPENROUTER_KEY || "";

let _sessionLang = null;
try { const s = localStorage.getItem("classio_lang"); _sessionLang = (s && s !== "auto") ? s : null; } catch {}

function _detectLang(text) {
  if (!text) return "English";
  if (/[\u0600-\u06FF]/.test(text)) return "Arabic";
  if (/[\u4E00-\u9FFF\u3040-\u30FF]/.test(text)) return "Chinese";
  if (/\b(le|la|les|de|est|et|je|tu|nous)\b/i.test(text)) return "French";
  if (/\b(el|la|los|es|en|que|por)\b/i.test(text)) return "Spanish";
  if (/\b(der|die|das|und|ist|ich)\b/i.test(text)) return "German";
  return "English";
}

async function _callOpenRouter(prompt, maxTok=3000) {
  if (!OPENROUTER_KEY) throw new Error("No OpenRouter key");
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method:"POST",
    headers:{ "Content-Type":"application/json", "Authorization":`Bearer ${OPENROUTER_KEY}`,
      "HTTP-Referer":"https://classio.app", "X-Title":"Classio" },
    body:JSON.stringify({ model:"mistralai/mistral-7b-instruct:free",
      messages:[{role:"user",content:prompt}], max_tokens:maxTok })
  });
  if (!res.ok) throw new Error("OpenRouter "+res.status);
  const d = await res.json();
  return d.choices?.[0]?.message?.content || "";
}

async function _callGeminiKey(key, prompt, maxTok) {
  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${key}`,
    { method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({ contents:[{parts:[{text:prompt}]}],
        generationConfig:{maxOutputTokens:maxTok,temperature:0.7} }) }
  );
  if (res.status===429) { const e=new Error("429"); e.status=429; throw e; }
  if (!res.ok) throw new Error("Gemini "+res.status);
  const d = await res.json();
  return d.candidates?.[0]?.content?.parts?.[0]?.text || "";
}

async function _callGemini(prompt, maxTok=3000) {
  for (const key of GEMINI_KEYS) {
    try { const t = await _callGeminiKey(key, prompt, maxTok); if (t) return t; }
    catch(e) { if (e.status===429) continue; throw e; }
  }
  try { return await _callOpenRouter(prompt, maxTok); } catch {}
  return await _callGroqWithRetry([{role:"user",content:prompt}], maxTok);
}

async function _callGroqWithRetry(messages, maxTok=3000, retries=2) {
  const models = ["llama-3.3-70b-versatile","llama-3.1-8b-instant","gemma2-9b-it"];
  for (const model of models) {
    for (let attempt=0; attempt<=retries; attempt++) {
      try {
        const res = await fetch(GROQ_URL, { method:"POST",
          headers:{"Content-Type":"application/json","Authorization":`Bearer ${GROQ_KEY}`},
          body:JSON.stringify({ model, messages, max_tokens: Math.min(maxTok, model.includes("8b")||model.includes("9b") ? 2000 : maxTok) }) });
        const d = await res.json();
        if (d.error) {
          const msg = d.error.message || "";
          if (msg.includes("Rate limit") || msg.includes("rate_limit") || res.status===429) {
            const waitMatch = msg.match(/try again in (\d+(?:\.\d+)?)s/i);
            const wait = waitMatch ? Math.min(parseFloat(waitMatch[1])*1000, 8000) : 3000 * (attempt+1);
            if (attempt < retries) { await new Promise(r=>setTimeout(r,wait)); continue; }
            else { break; }
          }
          throw new Error(msg);
        }
        return d.choices?.[0]?.message?.content || "";
      } catch(e) {
        if (attempt < retries && (e.message?.includes("Rate limit") || e.message?.includes("rate_limit"))) {
          await new Promise(r=>setTimeout(r,3000*(attempt+1)));
        } else if (attempt >= retries) { break; }
        else throw e;
      }
    }
  }
  throw new Error("All AI models are busy. Please wait a moment and try again.");
}

async function callClaude(system, userMessage, maxTok = 3000) {
  const prompt = (system ? system+"\n\n" : "") + userMessage;
  if (GEMINI_KEYS.length > 0) {
    try { return await _callGemini(prompt, maxTok); }
    catch(e) { console.warn("AI fallback:", e.message); }
  }
  return await _callGroqWithRetry(
    [{role:"system",content:system},{role:"user",content:userMessage}],
    maxTok
  );
}

// ─── AI IMAGE GENERATION (provider-agnostic) ─────────────────────────────────
const IMAGE_STYLE_PROMPTS = {
  realistic:   "photorealistic, high detail, natural lighting",
  illustration:"digital illustration, clean lines, vibrant colors",
  diagram:     "clean educational diagram, labeled, schematic, white background, vector style",
  infographic: "modern infographic style, clean icons, data visualization, organized layout",
  flat:        "flat design, simple shapes, minimal color palette, vector art",
  corporate:   "corporate professional style, clean, modern office aesthetic, blue and gray tones",
  modern:      "modern minimalist design, sleek, contemporary",
  startup:     "modern startup aesthetic, gradient colors, energetic, tech style",
  academic:    "academic textbook illustration style, precise, educational",
  futuristic:  "futuristic sci-fi style, neon accents, high-tech",
  minimal:     "minimalist, lots of negative space, simple shapes, muted colors",
  creative:    "creative artistic style, bold colors, expressive",
  render3d:    "3D render, soft shadows, studio lighting, CGI",
};

async function _pollinationsGenerate(prompt, style) {
  const styleHint = IMAGE_STYLE_PROMPTS[style] || "";
  const fullPrompt = styleHint ? `${prompt}, ${styleHint}` : prompt;
  const seed = Math.floor(Math.random() * 1e9);
  const url = `https://image.pollinations.ai/prompt/${encodeURIComponent(fullPrompt)}?width=1280&height=720&seed=${seed}&nologo=true`;
  await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = resolve;
    img.onerror = () => reject(new Error("Image generation failed"));
    img.src = url;
  });
  return url;
}

async function _openaiImageGenerate(prompt, style) { throw new Error("OpenAI image provider not configured"); }
async function _fluxGenerate(prompt, style) { throw new Error("Flux image provider not configured"); }
async function _ideogramGenerate(prompt, style) { throw new Error("Ideogram image provider not configured"); }
async function _recraftGenerate(prompt, style) { throw new Error("Recraft image provider not configured"); }

const IMAGE_PROVIDERS = {
  pollinations: _pollinationsGenerate,
  openai: _openaiImageGenerate,
  flux: _fluxGenerate,
  ideogram: _ideogramGenerate,
  recraft: _recraftGenerate,
};

async function generateAIImage(prompt, style = "diagram") {
  const providerName = process.env.REACT_APP_IMAGE_PROVIDER || "pollinations";
  const provider = IMAGE_PROVIDERS[providerName] || _pollinationsGenerate;
  try {
    const url = await provider(prompt, style);
    return { url, provider: providerName };
  } catch (e) {
    if (providerName !== "pollinations") {
      const url = await _pollinationsGenerate(prompt, style);
      return { url, provider: "pollinations" };
    }
    throw e;
  }
}

async function suggestImagePrompt({ topic, slideTitle, slideContent, isEducational = true }) {
  const raw = await callClaude(
    `You are a presentation design assistant. Given a slide's content, decide the single best visual to accompany it.
${isEducational ? "Prioritize diagrams, process visualizations, labeled illustrations, and infographics over generic photos — the image must help explain the concept, not just decorate." : ""}
Return ONLY valid JSON, no markdown:
{
  "prompt": "a clear, specific image generation prompt (1-2 sentences)",
  "style": "one of: realistic, illustration, diagram, infographic, flat, corporate, modern, startup, academic, futuristic, minimal, creative, render3d"
}`,
    `Presentation topic: "${topic}"\nSlide title: "${slideTitle}"\nSlide content: ${slideContent}\n\nSuggest the best image prompt and style for this slide.`,
    300
  );
  const clean = raw.replace(/```json|```/g, "").trim();
  const parsed = JSON.parse(clean.match(/\{[\s\S]*\}/)[0]);
  return { prompt: parsed.prompt || slideTitle, style: parsed.style || "diagram" };
}

async function callClaudeChat(system, messages) {
  const lastUser = [...messages].reverse().find(m=>m.role==="user");
  const raw = typeof lastUser?.content==="string" ? lastUser.content : "";
  const lang = _sessionLang || _detectLang(raw);
  const langEnforce = `\n\nCRITICAL: Respond ONLY in ${lang}. Do NOT mix languages. Do NOT translate unless explicitly asked.`;
  const sysP = (system||"") + langEnforce;

  if (GEMINI_KEYS.length > 0) {
    try {
      const hist = messages.map(m=>`${m.role==="user"?"User":"Assistant"}: ${typeof m.content==="string"?m.content:""}`).join("\n");
      return await _callGemini(sysP+"\n\n"+hist, 1200);
    } catch(e) { console.warn("Gemini chat fallback:", e.message); }
  }
  const res = await fetch(GROQ_URL, { method:"POST",
    headers:{"Content-Type":"application/json","Authorization":`Bearer ${GROQ_KEY}`},
    body:JSON.stringify({ model:"llama-3.3-70b-versatile",
      messages:[{role:"system",content:sysP},...messages], max_tokens:1200 }) });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.choices?.[0]?.message?.content || "";
}

async function callClaudeVision(system, messages, imageBase64) {
  const msgsWithImage = messages.map((m, i) => {
    if (i === messages.length - 1 && m.role === "user" && imageBase64) {
      return {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: imageBase64 } },
          { type: "text",      text: m.content || "Analyze this image and answer the question." },
        ],
      };
    }
    return m;
  });
  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "Authorization": `Bearer ${GROQ_KEY}` },
    body: JSON.stringify({
      model: "meta-llama/llama-4-scout-17b-16e-instruct",
      messages: [{ role: "system", content: system + "\n\nIMPORTANT: Always reply in the SAME language the user wrote in. Match exactly." }, ...msgsWithImage],
      max_tokens: 2000,
    }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.choices?.[0]?.message?.content || "";
}

// ─── GLOBAL VOICE SYSTEM ──────────────────────────────────────────────────────
const GLOBAL_PERSONAS = [
  { id:"aria",    label:"Aria",    gender:"female", color:"#92400e", desc:"Warm & conversational",  pitch:1.02, rate:0.95,
    targets:["microsoft aria online","aria online","jenny online","microsoft jenny online","zira","google us english","samantha","karen","victoria","fiona"] },
  { id:"nova",    label:"Nova",    gender:"female", color:"#7c3aed", desc:"Bright & upbeat",         pitch:1.08, rate:0.97,
    targets:["microsoft ava online","ava online","microsoft amber online","amber online","michelle online","microsoft michelle online","moira","google uk english female","tessa","kate"] },
  { id:"sage",    label:"Sage",    gender:"female", color:"#ca8a04", desc:"Calm & clear",             pitch:0.96, rate:0.92,
    targets:["microsoft emma online","emma online","microsoft sara online","sara online","microsoft jane online","jane online","fiona","google uk english female","karen","victoria"] },
  { id:"luna",    label:"Luna",    gender:"female", color:"#4f46e5", desc:"Soft & soothing",          pitch:1.0,  rate:0.90,
    targets:["microsoft ashley online","ashley online","microsoft ana online","ana online","siri","google us english","samantha","veena","allison","ting-ting"] },
  { id:"echo",    label:"Echo",    gender:"male",   color:"#2563eb", desc:"Steady & professional",   pitch:0.98, rate:0.93,
    targets:["microsoft guy online","guy online","microsoft eric online","eric online","microsoft davis online","davis online","daniel","google uk english male","alex","mark"] },
  { id:"onyx",    label:"Onyx",    gender:"male",   color:"#111827", desc:"Deep & authoritative",     pitch:0.88, rate:0.90,
    targets:["microsoft christopher online","christopher online","microsoft roger online","roger online","microsoft steffan online","steffan online","fred","google uk english male","lee","tom"] },
  { id:"fable",   label:"Fable",   gender:"male",   color:"#16a34a", desc:"Friendly & casual",        pitch:1.04, rate:0.95,
    targets:["microsoft ryan online","ryan online","microsoft liam online","liam online","microsoft noah online","noah online","rishi","thomas","oliver","google uk english male"] },
  { id:"atlas",   label:"Atlas",   gender:"male",   color:"#ea580c", desc:"Bold & energetic",         pitch:0.94, rate:1.00,
    targets:["microsoft brian online","brian online","microsoft reed online","reed online","microsoft andrew online","andrew online","alex","david","mark","google us english"] },
  { id:"river",   label:"River",   gender:"neutral",color:"#0891b2", desc:"Smooth & neutral",         pitch:1.0,  rate:0.93,
    targets:["microsoft jenny online","jenny online","microsoft guy online","guy online","google us english","google uk english","default","en-us","en-gb"] },
];

function renderAIText(text) {
  if (!text) return "";
  let t = text;
  t = t.replace(/\\\[[\s\S]*?\\\]/g, (m) => " " + m.slice(2, -2).trim() + " ");
  t = t.replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => " " + m.trim() + " ");
  t = t.replace(/\$([^$\n]+?)\$/g, (_, m) => m.trim());
  t = t.replace(/\\\([\s\S]*?\\\)/g, (m) => m.slice(2, -2).trim());
  t = t.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1)/($2)');
  t = t.replace(/\\sqrt\{([^}]+)\}/g, '√($1)');
  t = t.replace(/\\sqrt\b/g, '√');
  t = t.replace(/\^\{([^}]+)\}/g, '^($1)');
  t = t.replace(/\_\{([^}]+)\}/g, '_($1)');
  t = t.replace(/\\cdot/g, '·');
  t = t.replace(/\\times/g, '×');
  t = t.replace(/\\div/g, '÷');
  t = t.replace(/\\pm/g, '±');
  t = t.replace(/\\leq/g, '≤');
  t = t.replace(/\\geq/g, '≥');
  t = t.replace(/\\neq/g, '≠');
  t = t.replace(/\\approx/g, '≈');
  t = t.replace(/\\infty/g, '∞');
  t = t.replace(/\\pi/g, 'π');
  t = t.replace(/\\alpha/g, 'α');
  t = t.replace(/\\beta/g, 'β');
  t = t.replace(/\\theta/g, 'θ');
  t = t.replace(/\\Delta/g, 'Δ');
  t = t.replace(/\\delta/g, 'δ');
  t = t.replace(/\\sum/g, 'Σ');
  t = t.replace(/\\int/g, '∫');
  t = t.replace(/\\[a-zA-Z]+/g, '');
  t = t.replace(/[{}]/g, '');
  t = t.replace(/\*\*([^*]+)\*\*/g, '$1');
  t = t.replace(/\*([^*]+)\*/g, '$1');
  t = t.replace(/__([^_]+)__/g, '$1');
  t = t.replace(/_([^_]+)_/g, '$1');
  t = t.replace(/^#{1,6}\s+/gm, '');
  t = t.replace(/  +/g, ' ').trim();
  return t;
}

function getSmartVoice(personaOrIdx, allVoices, lang = "en-US") {
  const persona = typeof personaOrIdx === "number" ? GLOBAL_PERSONAS[personaOrIdx] : personaOrIdx;
  if (!persona || !allVoices || !allVoices.length) return null;

  const langCode = lang.slice(0, 2).toLowerCase();
  const langPool = allVoices.filter(v => v.lang.toLowerCase().startsWith(langCode));
  const pool = langPool.length > 0 ? langPool : allVoices;

  for (const tgt of persona.targets) {
    const v = pool.find(v => v.name.toLowerCase().includes(tgt.toLowerCase()));
    if (v) return v;
  }

  const femaleHints = /aria|jenny|zira|samantha|karen|victoria|moira|fiona|ava|emma|sara|ashley|tessa|kate|allison|susan|heather/i;
  const maleHints   = /guy|eric|david|mark|daniel|alex|fred|christopher|roger|brian|ryan|liam|reed|rishi|thomas|oliver/i;

  if (persona.gender === "female") {
    const v = pool.find(v => femaleHints.test(v.name));
    if (v) return v;
  } else if (persona.gender === "male") {
    const v = pool.find(v => maleHints.test(v.name));
    if (v) return v;
  }

  return (
    pool.find(v => /microsoft.*online/i.test(v.name)) ||
    pool.find(v => /google/i.test(v.name)) ||
    pool[0] || allVoices[0] || null
  );
}

function getSmartVoiceLabel(personaIdx, allVoices, lang = "en-US") {
  const v = getSmartVoice(personaIdx, allVoices, lang);
  if (!v) return "Default";
  return v.name
    .replace(/microsoft\s*/i, "")
    .replace(/\s*online.*$/i, "")
    .replace(/\s*-.*$/, "")
    .trim() || v.name;
}

// ─── AI DISTRACTOR GENERATOR ─────────────────────────────────────────────────
async function buildAIOptions(cards) {
  const subset = cards.slice(0, 40);
  const payload = subset.map(c => ({ id: c.id, q: c.question, a: c.answer }));

  const SYSTEM = `You are an expert quiz designer specialised in making believable wrong answers.
Your ONLY output must be a valid JSON array. No markdown, no explanation, nothing else.`;

  const USER = `For each item below, write exactly 3 DISTRACTOR answers (wrong but plausible).

STRICT RULES — follow every one:
1. Every distractor must be about the SAME specific concept as the correct answer.
2. Distractors must sound like they COULD be correct — someone who hasn't studied should struggle to choose.
3. Distractors must NOT copy any phrase from the correct answer.
4. All 4 options (correct + 3 distractors) must be similar in length and style.
5. Use the same vocabulary register (technical/simple) as the correct answer.
6. Do NOT use the exact question keyword as the answer — force the student to understand.

ITEMS:
${JSON.stringify(payload)}

Respond with ONLY this JSON array (no markdown fences):
[{"id":"<same id as input>","distractors":["wrong1","wrong2","wrong3"]}, ...]`;

  try {
    const raw = await callClaude(SYSTEM, USER, 4000);
    const clean = raw.replace(/```json|```/g, '').trim();
    const parsed = JSON.parse(clean);

    const map = new Map();
    for (const item of parsed) {
      const card = cards.find(c => String(c.id) === String(item.id));
      if (!card || !Array.isArray(item.distractors) || item.distractors.length < 3) continue;
      const four = [card.answer, ...item.distractors.slice(0, 3)].sort(() => Math.random() - .5);
      map.set(card.id, four);
    }

    for (const card of cards) {
      if (!map.has(card.id)) {
        const fallback = buildFallbackOptions(card, cards);
        map.set(card.id, fallback);
      }
    }
    return map;
  } catch (e) {
    console.warn('buildAIOptions failed, using fallback:', e);
    const map = new Map();
    for (const card of cards) map.set(card.id, buildFallbackOptions(card, cards));
    return map;
  }
}

function buildFallbackOptions(card, cards) {
  const others = cards.filter(x => x.id !== card.id).sort(() => Math.random() - .5).slice(0, 3).map(x => x.answer);
  while (others.length < 3) others.push('None of the above');
  return [card.answer, ...others].sort(() => Math.random() - .5);
}

// ─── MATH FORMATTER ──────────────────────────────────────────────────────────
function fixLanguage(text, langCode) {
  if (!text || !langCode) return text;
  const isArabic    = /[\u0600-\u06FF]/;
  const isChinese   = /[\u4E00-\u9FFF\u3400-\u4DBF]/;
  const isJapanese  = /[\u3040-\u309F\u30A0-\u30FF]/;
  const isKorean    = /[\uAC00-\uD7AF\u1100-\u11FF]/;
  const isCyrillic  = /[\u0400-\u04FF]/;
  const isHindi     = /[\u0900-\u097F]/;

  if (langCode.startsWith("ar")) {
    return text.split("").filter(c => {
      if (isChinese.test(c) || isJapanese.test(c) || isKorean.test(c) || isCyrillic.test(c) || isHindi.test(c)) return false;
      return true;
    }).join("");
  }
  if (langCode.startsWith("en")) {
    return text.split("").filter(c => {
      if (isArabic.test(c) || isChinese.test(c) || isJapanese.test(c) || isKorean.test(c) || isCyrillic.test(c) || isHindi.test(c)) return false;
      return true;
    }).join("");
  }
  if (langCode.startsWith("zh")) {
    return text.split("").filter(c => {
      if (isArabic.test(c) || isKorean.test(c) || isCyrillic.test(c) || isHindi.test(c)) return false;
      return true;
    }).join("");
  }
  return text;
}

function fixMath(text) {
  if (!text) return text;
  let t = text;

  const sup = { '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','-':'⁻','+':'⁺' };
  const toSup = (s) => String(s).split('').map(c => sup[c] || c).join('');

  const wordNum = {
    'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
    'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,
    'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,
    'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,
    'negative':'-','minus':'-','plus':'+'
  };

  const parseWordNum = (s) => {
    s = s.toLowerCase().trim();
    if (!isNaN(s)) return s;
    if (wordNum[s] !== undefined) return String(wordNum[s]);
    const parts = s.split(/\s+/);
    if (parts.length === 2 && (parts[0]==='negative'||parts[0]==='minus')) {
      const n = wordNum[parts[1]];
      if (n !== undefined) return String(-n);
    }
    if (parts.length === 2 && parts[0]==='positive') {
      const n = wordNum[parts[1]];
      if (n !== undefined) return String(n);
    }
    return null;
  };

  t = t.replace(/\btimes\b/gi, '×');
  t = t.replace(/\bmultiplied by\b/gi, '×');

  t = t.replace(/\bto the power of\s+(negative|minus|positive)?\s*([\w]+)/gi, (m, sign, num) => {
    const s = sign ? (sign[0]==='p'?'+':'-') : '';
    const n = parseWordNum(num) ?? num;
    return toSup(s + n);
  });
  t = t.replace(/\bto the\s+(negative|minus)?\s*([\w]+)\s*power/gi, (m, sign, num) => {
    const s = sign ? '-' : '';
    const n = parseWordNum(num) ?? num;
    return toSup(s + n);
  });
  t = t.replace(/\braised to\s+(negative|minus|positive)?\s*([\w]+)/gi, (m, sign, num) => {
    const s = sign ? (sign[0]==='p'?'+':'-') : '';
    const n = parseWordNum(num) ?? num;
    return toSup(s + n);
  });
  t = t.replace(/\^([+-]?\d+)/g, (m, exp) => toSup(exp));
  t = t.replace(/(\d+(?:\.\d+)?)[eE]([+-]?\d+)/g, (m, base, exp) => `${base} × 10${toSup(exp)}`);

  const numWords = Object.keys(wordNum).join('|');
  t = t.replace(
    new RegExp(`\\b(${numWords})\\s+point\\s+(${numWords})\\b`, 'gi'),
    (m, a, b) => {
      const av = wordNum[a.toLowerCase()];
      const bv = wordNum[b.toLowerCase()];
      if (av !== undefined && bv !== undefined) return `${av}.${bv}`;
      return m;
    }
  );

  t = t.replace(
    new RegExp(`\\b(${numWords})\\s+(×)`, 'gi'),
    (m, num, op) => {
      const v = wordNum[num.toLowerCase()];
      return v !== undefined ? `${v} ${op}` : m;
    }
  );

  t = t.replace(/\b(\d[\d.,]*)\s*centimetre[s]?\b/gi, '$1 cm');
  t = t.replace(/\b(\d[\d.,]*)\s*metre[s]?\b/gi, '$1 m');
  t = t.replace(/\b(\d[\d.,]*)\s*kilometre[s]?\b/gi, '$1 km');
  t = t.replace(/\b(\d[\d.,]*)\s*millimetre[s]?\b/gi, '$1 mm');
  t = t.replace(/\b(\d[\d.,]*)\s*nanometre[s]?\b/gi, '$1 nm');
  t = t.replace(/\b(\d[\d.,]*)\s*kilogram[s]?\b/gi, '$1 kg');
  t = t.replace(/\b(\d[\d.,]*)\s*gram[s]?\b/gi, '$1 g');
  t = t.replace(/\b(\d[\d.,]*)\s*milligram[s]?\b/gi, '$1 mg');
  t = t.replace(/\b(\d[\d.,]*)\s*second[s]?\b/gi, (m,n) => `${n} s`);
  t = t.replace(/\b(\d[\d.,]*)\s*millisecond[s]?\b/gi, '$1 ms');
  t = t.replace(/\b(\d[\d.,]*)\s*joule[s]?\b/gi, '$1 J');
  t = t.replace(/\b(\d[\d.,]*)\s*newton[s]?\b/gi, '$1 N');
  t = t.replace(/\b(\d[\d.,]*)\s*watt[s]?\b/gi, '$1 W');
  t = t.replace(/\b(\d[\d.,]*)\s*volt[s]?\b/gi, '$1 V');
  t = t.replace(/\b(\d[\d.,]*)\s*ampere[s]?|amp[s]?\b/gi, '$1 A');
  t = t.replace(/\b(\d[\d.,]*)\s*pascal[s]?\b/gi, '$1 Pa');
  t = t.replace(/\b(\d[\d.,]*)\s*kelvin[s]?\b/gi, '$1 K');

  t = t.replace(/\bone half\b/gi, '1/2');
  t = t.replace(/\bone third\b/gi, '1/3');
  t = t.replace(/\bone quarter\b/gi, '1/4');
  t = t.replace(/\bthree quarter[s]?\b/gi, '3/4');
  t = t.replace(/\btwo third[s]?\b/gi, '2/3');

  t = t.replace(/\bsquared\b/gi, '²');
  t = t.replace(/\bcubed\b/gi, '³');
  t = t.replace(/\bsquare root of\b/gi, '√');

  t = t.replace(/\bdivided by\b/gi, '÷');
  t = t.replace(/\bover\b(?=\s+[\d(])/gi, '/');

  t = t.replace(/\bapproximately equal[s]?\s+to\b/gi, '≈');
  t = t.replace(/\bapproximately\b/gi, '≈');
  t = t.replace(/\bgreater than or equal[s]?\s+to\b/gi, '≥');
  t = t.replace(/\bless than or equal[s]?\s+to\b/gi, '≤');
  t = t.replace(/\bgreater than\b/gi, '>');
  t = t.replace(/\bless than\b/gi, '<');
  t = t.replace(/\bnot equal[s]?\s+to\b/gi, '≠');
  t = t.replace(/\bplus or minus\b/gi, '±');

  t = t.replace(/\bdegree[s]?\b(?!\s*[CF])/gi, '°');
  t = t.replace(/\bdegree[s]?\s+Celsius\b/gi, '°C');
  t = t.replace(/\bdegree[s]?\s+Fahrenheit\b/gi, '°F');
  t = t.replace(/\bdegree[s]?\s+Kelvin\b/gi, 'K');

  t = t.replace(/\balpha\b/gi, 'α'); t = t.replace(/\bbeta\b/gi, 'β');
  t = t.replace(/\bgamma\b/gi, 'γ'); t = t.replace(/\bdelta\b/gi, 'Δ');
  t = t.replace(/\blambda\b/gi, 'λ'); t = t.replace(/\bmu\b/gi, 'μ');
  t = t.replace(/\bpi\b(?!\s*[a-z])/gi, 'π'); t = t.replace(/\bsigma\b/gi, 'σ');
  t = t.replace(/\bomega\b/gi, 'ω'); t = t.replace(/\btheta\b/gi, 'θ');
  t = t.replace(/\bepsilon\b/gi, 'ε'); t = t.replace(/\bphi\b/gi, 'φ');
  t = t.replace(/\binfinity\b/gi, '∞');

  return t;
}

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

async function extractFileText(fileObj) {
  if (!fileObj) return null;
  const name = fileObj.name.toLowerCase();
  const type = fileObj.type || "";

  if (type.startsWith("text/") || name.endsWith(".txt") || name.endsWith(".md") || name.endsWith(".csv")) {
    try { return await readFileAsText(fileObj); } catch { return null; }
  }

  if (type === "application/pdf" || name.endsWith(".pdf")) {
    try {
      const base64 = await readFileAsBase64(fileObj);
      const binary = atob(base64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

      if (!window.pdfjsLib) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
        window.pdfjsLib.GlobalWorkerOptions.workerSrc =
          "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      }

      const pdf = await window.pdfjsLib.getDocument({ data: bytes }).promise;
      let text = "";
      for (let i = 1; i <= pdf.numPages; i++) {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        text += content.items.map(item => item.str).join(" ") + "\n";
      }
      return text.trim() || null;
    } catch(e) { console.error("PDF read error", e); return null; }
  }

  if (type.startsWith("image/")) {
    return `[This is an image file: ${fileObj.name}. Describe its likely content based on the filename.]`;
  }

  if (name.endsWith(".pptx") || name.endsWith(".docx") || name.endsWith(".xlsx") ||
      name.endsWith(".ppt") || name.endsWith(".doc") || name.endsWith(".xls")) {
    try {
      if (!window.JSZip) {
        await new Promise((res, rej) => {
          const s = document.createElement("script");
          s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
      }
      const arrayBuffer = await fileObj.arrayBuffer();
      const zip = await window.JSZip.loadAsync(arrayBuffer);
      let text = "";

      const xmlFiles = Object.keys(zip.files).filter(f =>
        f.endsWith(".xml") && (
          f.includes("slide") || f.includes("word/document") ||
          f.includes("sharedStrings") || f.includes("content")
        )
      );

      for (const xmlFile of xmlFiles) {
        const xmlContent = await zip.files[xmlFile].async("string");
        const stripped = xmlContent.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
        if (stripped.length > 20) text += stripped + "\n";
      }

      return text || null;
    } catch(e) { console.error("Office file read error", e); return null; }
  }

  return null;
}

// ─── RESPONSIVE HOOK ─────────────────────────────────────────────────────────
function useResponsive() {
  const getState = () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const isLandscape = w > h;
    const shortEdge = Math.min(w, h);
    const longEdge  = Math.max(w, h);

    const ua = navigator.userAgent || "";
    const isIOS     = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const isAndroid = /Android/.test(ua);
    const isTouchUA = isIOS || isAndroid || /Mobi|Tablet/.test(ua);

    const isPhone = shortEdge <= 480 || (shortEdge <= 600 && isTouchUA && longEdge <= 900);
    const isTabletDevice = !isPhone && (shortEdge <= 1024) &&
      (isTouchUA || (shortEdge <= 900 && longEdge <= 1400));
    const isLargeTablet = isTabletDevice && shortEdge >= 900;

    const size = isPhone ? "phone" : isTabletDevice ? "tablet" : "desktop";

    return { size, isLandscape, w, h, shortEdge, longEdge, isIOS, isAndroid, isTouchUA, isLargeTablet };
  };

  const [state, setState] = useState(getState);

  useEffect(() => {
    let raf;
    const fn = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setState(getState()));
    };
    window.addEventListener("resize", fn, { passive: true });
    window.addEventListener("orientationchange", fn, { passive: true });
    window.screen?.orientation?.addEventListener("change", fn);
    return () => {
      window.removeEventListener("resize", fn);
      window.removeEventListener("orientationchange", fn);
      window.screen?.orientation?.removeEventListener("change", fn);
      cancelAnimationFrame(raf);
    };
  }, []);

  const { size, isLandscape, w, h, isIOS, isAndroid, isTouchUA, isLargeTablet } = state;
  const isMobile   = size === "phone";
  const isTablet   = size === "tablet";
  const isDesktop  = size === "desktop";
  const isPhoneLandscape  = isMobile && isLandscape;
  const isTabletLandscape = isTablet && isLandscape;
  const isTouchDevice     = isMobile || isTablet || isTouchUA;

  const sidebarW      = isMobile ? 0 : isTablet ? 60 : 220;
  const contentPadX   = isMobile ? 12 : isTablet ? 20 : 32;
  const contentPadY   = isMobile ? 12 : isTablet ? 16 : 24;
  const fontScale     = isMobile ? 0.88 : isTablet ? 0.94 : 1;
  const notesSplit    = isMobile ? "100%" : isTablet ? (isLandscape ? "60%" : "100%") : "63%";
  const showAIPanel   = !isMobile && !(isTablet && !isLandscape);
  const quickActCols  = isMobile ? "1fr 1fr" : isTablet ? "1fr 1fr" : "repeat(4,1fr)";

  return {
    isMobile, isTablet, isDesktop, size, isLandscape,
    isPhoneLandscape, isTabletLandscape, isTouchDevice,
    isIOS, isAndroid, isLargeTablet,
    w, h,
    sidebarW, contentPadX, contentPadY, fontScale,
    notesSplit, showAIPanel, quickActCols,
  };
}

// ─── THEME SYSTEM ─────────────────────────────────────────────────────────────
const GRAD = "linear-gradient(135deg,#7C5CFC 0%,#3D8EF8 100%)";

const THEMES = {
  light: {
    bg:"#F8F7F4", surface:"#FFFFFF", border:"#EAE6E1", text:"#1A1714",
    muted:"#9B9590", accent:"#6C5CE7", accentL:"#F0EEFF", accentS:"#D4CCFF",
    warm:"#C17F5A", warmL:"#F5EDE5", green:"#4A7C59", greenL:"#E5F0E8",
    purple:"#7C5CFC", purpleL:"#EDE5F5", red:"#C45C5C", redL:"#F5E5E5",
    sidebar:"#FFFFFF", sidebarActive:"#F3F1FF", navText:"#1A1714",
    shadow:"0 2px 16px rgba(108,92,231,.07)", cardShadow:"0 4px 24px rgba(108,92,231,.10)",
    grad: GRAD,
    isDark:false,
  },
  dark: {
    bg:"#1A1A15", surface:"#252520", border:"#2E2E28", text:"#EEECEA",
    muted:"#6A6A62", accent:"#7C5CFC", accentL:"#1E1830", accentS:"#2D2050",
    warm:"#C17F5A", warmL:"#2A1F16", green:"#5A9C69", greenL:"#162218",
    purple:"#9B7ECA", purpleL:"#261A35", red:"#D47070", redL:"#2A1616",
    sidebar:"#131310", sidebarActive:"#252520", navText:"#EEECEA",
    shadow:"0 2px 12px rgba(0,0,0,.6)", cardShadow:"0 4px 20px rgba(0,0,0,.55)",
    grad: GRAD,
    isDark:true,
  },
};
let _isDark = false;
try { _isDark = localStorage.getItem("classio_dark")==="true"; } catch {}
if (_isDark && typeof document !== "undefined") { document.body.classList.add("classio-dark"); }
let _themeCallbacks = [];
function _getTheme() { return _isDark ? THEMES.dark : THEMES.light; }
function _toggleTheme() {
  _isDark = !_isDark;
  try { localStorage.setItem("classio_dark", String(_isDark)); } catch {}
  document.body.style.background = _getTheme().bg;
  if (_isDark) document.body.classList.add("classio-dark"); else document.body.classList.remove("classio-dark");
  _themeCallbacks.forEach(fn => fn());
}
function useTheme() {
  const [t, setT] = useState(() => _getTheme());
  useEffect(() => {
    const fn = () => setT(_getTheme());
    _themeCallbacks.push(fn);
    return () => { _themeCallbacks = _themeCallbacks.filter(f=>f!==fn); };
  }, []);
  return t;
}
let C = _getTheme();

const FILE_COLORS = [
  { bg:"#E8EFF5", accent:"#3D5A80" }, { bg:"#F5EDE5", accent:"#C17F5A" },
  { bg:"#E5F0E8", accent:"#4A7C59" }, { bg:"#EDE5F5", accent:"#6B4E8A" },
  { bg:"#F5E5E5", accent:"#C45C5C" }, { bg:"#EAEDF0", accent:"#4A5568" },
  { bg:"#FFF8E1", accent:"#D69E2E" }, { bg:"#E0F7FA", accent:"#0694a2" },
  { bg:"#FCE4EC", accent:"#E91E8C" }, { bg:"#F3E5F5", accent:"#7B1FA2" },
  { bg:"#E8F5E9", accent:"#2E7D32" }, { bg:"#FBE9E7", accent:"#BF360C" },
];

function getFileColor(file) {
  if (file.customColor) {
    const c = file.customColor;
    return { accent: c, bg: c + "22" };
  }
  return FILE_COLORS[file.colorIndex||0] || FILE_COLORS[0];
}
const FOLDER_COLORS = ["#3D5A80","#C17F5A","#4A7C59","#6B4E8A","#C45C5C","#4A5568","#8A7C4E","#4E7C8A"];

// ─── ICONS ────────────────────────────────────────────────────────────────────
const Icon = ({ d, size = 18, color = "currentColor", sw = 1.7 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {Array.isArray(d) ? d.map((p, i) => <path key={i} d={p} />) : <path d={d} />}
  </svg>
);
const I = {
  folder: "M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z",
  file: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6"],
  plus: "M12 5v14M5 12h14",
  back: "M19 12H5M12 19l-7-7 7-7",
  ai: "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zM8 14s1.5 2 4 2 4-2 4-2M9 9h.01M15 9h.01",
  cards: "M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 1-4 4v14a3 3 0 0 0 3-3h7z",
  notes: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6","M16 13H8","M16 17H8","M10 9H8"],
  game: "M6 3v11.5A2.5 2.5 0 0 0 8.5 17h.5M18 3v11.5A2.5 2.5 0 0 1 15.5 17h-.5M8.5 17a2.5 2.5 0 0 0 0 5 2.5 2.5 0 0 0 0-5zM15.5 17a2.5 2.5 0 0 0 0 5 2.5 2.5 0 0 0 0-5z",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  send: "M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z",
  trash: "M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2",
  edit: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z",
  check: "M20 6L9 17l-5-5",
  sparkle: "M12 3L14.5 8.5H20L15.5 12L17 18L12 14.5L7 18L8.5 12L4 8.5H9.5L12 3Z",
  link: "M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71",
  x: "M18 6L6 18M6 6l12 12",
  chevron: "M9 18l6-6-6-6",
  refresh: "M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15",
  paperclip: "M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48",
  mic: "M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3zM19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8",
  headphones: "M3 18v-6a9 9 0 0 1 18 0v6M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z",
  globe: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zM2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z",
  robot: "M12 2a2 2 0 0 1 2 2v1h3a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h3V4a2 2 0 0 1 2-2zM9 11h.01M15 11h.01M9 15s1 1.5 3 1.5 3-1.5 3-1.5",
  zap: "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
  star: "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
  users: "M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 7a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75",
  image: "M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2zM8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM21 15l-5-5L5 19",
  pen: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z",
  eraser: "M20 20H7L3 16l10-10 7 7-3.5 3.5M6.5 17.5l5-5",
  highlight: "M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z",
  podcast: "M8.56 2.9A7 7 0 0 1 19 9v4M2 9a10 10 0 0 1 20 0v5a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2zM6 9v1a6 6 0 0 0 12 0V9M12 16v6M8 22h8",
  info: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 8h.01M11 12h1v4h1",
  regenerate: "M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15",
  atom: "M12 12m-1 0a1 1 0 1 0 2 0 1 1 0 1 0-2 0M20.2 20.2c2.04-2.03.02-7.36-4.5-11.9-4.54-4.52-9.87-6.54-11.9-4.5-2.04 2.03-.02 7.36 4.5 11.9 4.54 4.52 9.87 6.54 11.9 4.5zM15.7 15.7c4.52-4.54 6.54-9.87 4.5-11.9-2.03-2.04-7.36-.02-11.9 4.5-4.52 4.54-6.54 9.87-4.5 11.9 2.03 2.04 7.36.02 11.9-4.5z",
  satellite: "M13 7l4 4L9 19l-4-4 8-8zM5 16l-2 2 4 4 2-2M14 6l3-3 5 5-3 3M9 9l1 1M19 14l1 1",
  flask: "M9 2h6M10 2v6.5L4.5 18A2 2 0 0 0 6 21h12a2 2 0 0 0 1.5-3L14 8.5V2M8.5 14h7",
  ruler: "M16 3l5 5L7 22l-5-5zM14.5 4.5l3 3M12 7l1.5 1.5M9 10l1.5 1.5M6 13l1.5 1.5",
  telescope: "M10.5 9L3 14l1.5 2.5L12 12M10.5 9L21 6l-3 9-7.5-3.5M10.5 9l1.5 4.5M6 22l3-7M14 13l4 7",
  dna: "M9 2c0 4-6 4-6 8s6 4 6 8M15 2c0 4 6 4 6 8s-6 4-6 8M5 7h14M5 17h14",
  chart: "M3 3v18h18M7 17V10M12 17V7M17 17v-5",
  bolt: "M13 2L3 14h9l-1 8 10-12h-9l1-8z",
  wave: "M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0 4 4 6 0",
  battery: "M2 7h16v10H2zM18 10h2v4h-2zM6 10v4M10 10v4M14 10v4",
  magnet: "M6 15v-3a6 6 0 0 1 12 0v3M6 15H2v4a2 2 0 0 0 2 2h2zM18 15h4v4a2 2 0 0 1-2 2h-2zM6 15v6M18 15v6",
  lightbulb: "M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z",
  key: "M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4",
  pin: "M12 17v5M9 10.76a2 2 0 0 1-.76 1.69l-.95.81A2 2 0 0 0 6.5 15h11a2 2 0 0 0-.79-1.74l-.95-.81A2 2 0 0 1 15 10.76V7h1V5H8v2h1z",
  microscope: "M6 18h8M3 22h18M14 22a7 7 0 1 0 0-14h-1M9 14h2M9 12a2 2 0 0 1-2-2V8a2 2 0 0 1 4 0v4a2 2 0 0 1-2 2zM12 6.5V4h2",
  bookOpen: "M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2zM22 3h-6a4 4 0 0 1-4 4v14a3 3 0 0 0 3-3h7z",
  gradCap: "M22 10v6M2 10l10-5 10 5-10 5z M6 12v5c0 1.1 2.7 2 6 2s6-.9 6-2v-5",
  video: "M23 7l-7 5 7 5V7zM14 5H3a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2z",
  fileText: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6","M16 13H8","M16 17H8","M10 9H8"],
  presentation: "M3 3h18M9 21l3-6 3 6M3 3v12a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V3",
  imageFile: "M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2zM8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM21 15l-5-5L5 19",
  audioFile: "M9 18V5l12-2v13M9 18a3 3 0 1 0 0 0zM21 16a3 3 0 1 0 0 0z",
  fileGeneric: ["M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z","M14 2v6h6"],
  trophy: "M6 9H4.5a2.5 2.5 0 0 1 0-5H6M18 9h1.5a2.5 2.5 0 0 0 0-5H18M6 4h12v6a6 6 0 0 1-12 0V4zM10 16v2M14 16v2M8 22h8M9 18h6v2a1 1 0 0 1-1 1H10a1 1 0 0 1-1-1v-2z",
  messageCircle: "M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z",
};

// ─── TEXT FORMATTER ───────────────────────────────────────────────────────────
function renderMathInline(latex) {
  try {
    if (window.katex) {
      return window.katex.renderToString(latex, { throwOnError: false, displayMode: false });
    }
  } catch(e) {}
  return latex;
}

function renderMathDisplay(latex) {
  try {
    if (window.katex) {
      return window.katex.renderToString(latex, { throwOnError: false, displayMode: true });
    }
  } catch(e) {}
  return latex;
}

function parseMathSegments(text) {
  if (!text) return [{ type: "text", content: "" }];
  const segments = [];
  const mathRe = /(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$[^$\n]{1,300}\$|\\\([\s\S]*?\\\))/g;
  let last = 0;
  let match;
  while ((match = mathRe.exec(text)) !== null) {
    if (match.index > last) {
      segments.push({ type: "text", content: text.slice(last, match.index) });
    }
    const raw = match[0];
    const isDisplay = raw.startsWith("$$") || raw.startsWith("\\[");
    let latex = raw;
    if (raw.startsWith("$$")) latex = raw.slice(2, -2);
    else if (raw.startsWith("\\[")) latex = raw.slice(2, -2);
    else if (raw.startsWith("$")) latex = raw.slice(1, -1);
    else if (raw.startsWith("\\(")) latex = raw.slice(2, -2);
    segments.push({ type: isDisplay ? "display" : "inline", content: latex.trim() });
    last = match.index + raw.length;
  }
  if (last < text.length) segments.push({ type: "text", content: text.slice(last) });
  return segments;
}

function stripLatex(text) {
  if (!text) return "";
  let t = text;
  t = t.replace(/\$\$([\s\S]*?)\$\$/g, (_, m) => " " + m.trim() + " ");
  t = t.replace(/\$([^$\n]{1,200})\$/g, (_, m) => m.trim());
  t = t.replace(/\\\[[\s\S]*?\\\]/g, (m) => " " + m.slice(2, -2).trim() + " ");
  t = t.replace(/\\\([\s\S]*?\\\)/g, (m) => m.slice(2, -2).trim());
  t = t.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, '($1)/($2)');
  t = t.replace(/\\sqrt\{([^{}]+)\}/g, '√($1)');
  t = t.replace(/\^\{2\}/g, '²'); t = t.replace(/\^\{3\}/g, '³');
  t = t.replace(/\^\{([^{}]+)\}/g, '^($1)');
  t = t.replace(/\\cdot/g, '·'); t = t.replace(/\\times/g, '×');
  t = t.replace(/\\div/g, '÷'); t = t.replace(/\\pm/g, '±');
  t = t.replace(/\\leq/g, '≤'); t = t.replace(/\\geq/g, '≥');
  t = t.replace(/\\neq/g, '≠'); t = t.replace(/\\approx/g, '≈');
  t = t.replace(/\\infty/g, '∞'); t = t.replace(/\\pi/g, 'π');
  t = t.replace(/\\alpha/g, 'α'); t = t.replace(/\\beta/g, 'β');
  t = t.replace(/\\theta/g, 'θ'); t = t.replace(/\\Delta/g, 'Δ');
  t = t.replace(/\\sum/g, 'Σ'); t = t.replace(/\\int/g, '∫');
  t = t.replace(/\\to/g, '→'); t = t.replace(/\\rightarrow/g, '→');
  t = t.replace(/\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}/g, '$1');
  t = t.replace(/\\[a-zA-Z]+/g, ''); t = t.replace(/[{}]/g, '');
  t = t.replace(/  +/g, ' ').trim();
  return t;
}

function MathText({ text, style }) {
  const segments = parseMathSegments(text || "");
  const hasKatex = typeof window !== "undefined" && window.katex;
  if (!hasKatex) {
    return <span style={style}>{stripLatex(text || "")}</span>;
  }
  return (
    <span style={style}>
      {segments.map((seg, i) => {
        if (seg.type === "text") return <span key={i}>{seg.content}</span>;
        if (seg.type === "display") {
          const html = renderMathDisplay(seg.content);
          return <span key={i} style={{ display:"block", textAlign:"center", margin:"8px 0", overflowX:"auto" }}
            dangerouslySetInnerHTML={{ __html: html }} />;
        }
        const html = renderMathInline(seg.content);
        return <span key={i} dangerouslySetInnerHTML={{ __html: html }} />;
      })}
    </span>
  );
}

// ─── MERMAID DIAGRAM RENDERER ────────────────────────────────────────────────
function MermaidBlock({ code }) {
  const ref = useRef(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!code || !ref.current) return;
    const render = async () => {
      try {
        if (!window.mermaid) {
          await new Promise((res, rej) => {
            const s = document.createElement("script");
            s.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
            s.onload = res; s.onerror = rej;
            document.head.appendChild(s);
          });
          window.mermaid.initialize({ startOnLoad:false, theme:"default", securityLevel:"loose" });
        }
        const id = "mermaid-" + Math.random().toString(36).slice(2);
        const { svg } = await window.mermaid.render(id, code.trim());
        if (ref.current) ref.current.innerHTML = svg;
      } catch(e) {
        setError(true);
      }
    };
    render();
  }, [code]);

  if (error) return (
    <div style={{ background:C.warmL, border:`1px solid ${C.warm}33`, borderRadius:10, padding:"10px 14px", fontSize:12, color:C.muted }}>
      <p style={{ margin:0, fontWeight:600 }}>Diagram syntax error — raw code:</p>
      <pre style={{ margin:"6px 0 0", fontSize:11, overflow:"auto" }}>{code}</pre>
    </div>
  );

  return (
    <div style={{ background:C.surface, border:`1.5px solid ${C.accentS}`, borderRadius:14, padding:"16px", margin:"8px 0", overflow:"auto" }}>
      <p style={{ fontSize:10, fontWeight:700, color:C.accent, letterSpacing:.8, marginBottom:8 }}>📊 DIAGRAM</p>
      <div ref={ref} style={{ minHeight:60 }}/>
    </div>
  );
}

const Fmt = ({ text }) => {
  if (!text) return null;
  const lines = text.split('\n');
  const elements = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.trim().startsWith("```mermaid")) {
      const mermaidLines = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        mermaidLines.push(lines[i]);
        i++;
      }
      elements.push(<MermaidBlock key={`m${i}`} code={mermaidLines.join('\n')} />);
      i++;
      continue;
    }
    if (line.trim().startsWith("```")) {
      const codeLines = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        codeLines.push(lines[i]);
        i++;
      }
      elements.push(
        <pre key={`c${i}`} style={{ background:C.bg, border:`1px solid ${C.border}`, borderRadius:8, padding:"10px 14px", fontSize:12, overflowX:"auto", margin:"6px 0", color:C.text }}>
          {codeLines.join('\n')}
        </pre>
      );
      i++;
      continue;
    }
    const clean = line.replace(/^#+\s*/, "").replace(/\*\*/g, "").replace(/\*/g, "").trim();
    if (!line.trim()) { elements.push(<br key={i} />); i++; continue; }
    if (/^[A-Z][A-Z\s]{3,}$/.test(clean) || line.startsWith('# ') || line.startsWith('## '))
      { elements.push(<p key={i} style={{ fontWeight:700, fontSize:14, marginBottom:4, marginTop:12, color:C.text, letterSpacing:.3 }}><MathText text={clean} /></p>); i++; continue; }
    if (/^\*\*[^*]+\*\*$/.test(line.trim()))
      { elements.push(<p key={i} style={{ fontWeight:700, fontSize:14, marginBottom:4, marginTop:10, color:C.text }}><MathText text={clean} /></p>); i++; continue; }
    if (line.startsWith('• ') || line.startsWith('- ') || line.startsWith('* ') || line.startsWith('· '))
      { elements.push(<p key={i} style={{ paddingLeft:14, marginBottom:3, display:"flex", gap:6, lineHeight:1.6 }}><span style={{flexShrink:0, color:C.muted}}>•</span><span><MathText text={clean.replace(/^[•\-\*·]\s*/,"")} /></span></p>); i++; continue; }
    if (/^\d+\.\s/.test(line))
      { elements.push(<p key={i} style={{ paddingLeft:14, marginBottom:3, lineHeight:1.6 }}><MathText text={clean} /></p>); i++; continue; }
    elements.push(<p key={i} style={{ marginBottom:3, lineHeight:1.6 }}><MathText text={clean} /></p>);
    i++;
  }
  return <div>{elements}</div>;
};

// ─── GLOBAL STYLES ────────────────────────────────────────────────────────────
const GS = `@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Fraunces:wght@400;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0} input,textarea,button{font-family:inherit}
::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-thumb{background:#D8D4CF;border-radius:3px}
.hov:hover{opacity:0.82} .card-hov:hover{box-shadow:0 8px 24px rgba(0,0,0,.10)!important}
@keyframes sg-fadein{from{opacity:0;transform:translateX(-50%) translateY(-8px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}
.card-hov{transition:all .2s} .tab:hover{background:#F0EDE9!important} .row:hover{background:#F7F5F2!important} .row{transition:background .15s}

@media(max-width:600px){
  .desktop-only{display:none!important}
  .nav-tabs{overflow-x:auto!important;-webkit-overflow-scrolling:touch;scrollbar-width:none;flex-wrap:nowrap!important}
  .nav-tabs::-webkit-scrollbar{display:none}
  .nav-tab-btn{padding:10px 10px!important;font-size:12px!important;white-space:nowrap!important}
  .tab-label{display:none!important}
  .page-inner{padding:14px 12px!important}
  .page-with-ad{padding-bottom:60px!important}
  .app-header{padding:0 12px!important;min-height:52px!important}
  .card-grid{grid-template-columns:1fr 1fr!important}
  .game-grid{grid-template-columns:1fr 1fr!important}
  .mobile-stack{flex-direction:column!important}
  .mobile-full{width:100%!important;max-width:100%!important}
  .modal-inner{border-radius:18px 18px 0 0!important;max-height:92vh!important;width:100%!important;position:fixed!important;bottom:0!important;left:0!important;right:0!important;margin:0!important}
  button{min-height:40px}
  .chat-input-row{flex-wrap:wrap;gap:6px!important}
  .page-inner{overflow-x:hidden!important}
  button:not(.no-min-h){min-height:36px}
  .podcast-player{margin:0!important;border-radius:0!important}
  .card-label{overflow:hidden!important;text-overflow:ellipsis!important;white-space:nowrap!important;max-width:100%!important}
  .mobile-action-bar{display:flex!important}
  .modal-inner{max-height:88vh!important}
  .content-card{width:100%!important;max-width:100%!important;box-sizing:border-box!important}
  body{overflow-x:hidden!important}
  [style*="border-radius: 50%"],[style*="border-radius:50%"]{
    aspect-ratio:1!important;
    min-height:unset!important;
    min-width:unset!important;
    flex-shrink:0!important;
  }
  .color-swatch,.folder-color-btn{
    width:24px!important;
    height:24px!important;
    min-height:unset!important;
    min-width:unset!important;
    border-radius:50%!important;
    aspect-ratio:1!important;
    flex-shrink:0!important;
    padding:0!important;
  }
  .color-swatch{width:36px!important;height:36px!important}
}

@media(max-height:500px) and (orientation:landscape){
  .app-header{min-height:44px!important;height:44px!important}
  .modal-inner{max-height:90vh!important;border-radius:12px!important;position:relative!important;bottom:auto!important;margin:auto!important}
  .landscape-hide{display:none!important}
  .ad-banner-wrap{display:none!important}
  .page-with-ad{padding-bottom:0!important}
}

@media(max-width:900px) and (orientation:landscape){
  .ad-banner-wrap{display:none!important}
  .page-with-ad{padding-bottom:0!important}
}

@media(max-width:900px){
  .nav-tabs{overflow-x:auto;-webkit-overflow-scrolling:touch;scrollbar-width:none}
  .nav-tabs::-webkit-scrollbar{display:none}
}

@media(min-width:601px) and (max-width:900px) and (orientation:portrait){
  .notes-split{flex-direction:column!important}
  .notes-ai-panel{width:100%!important;min-width:unset!important;max-height:340px}
  .cards-grid{grid-template-columns:1fr 1fr!important}
  .game-grid{grid-template-columns:1fr 1fr 1fr!important}
  .quick-actions{grid-template-columns:1fr 1fr!important}
  .view-split{flex-direction:column!important}
  .view-ai-panel{width:100%!important;height:300px!important;border-left:none!important;border-top:1px solid var(--border)}
  .file-list-actions{flex-wrap:wrap}
}

@media(min-width:601px) and (max-width:1024px) and (orientation:landscape){
  .notes-split{gap:12px!important}
  .notes-ai-panel{width:260px!important;min-width:240px!important}
  .cards-grid{grid-template-columns:1fr 1fr 1fr!important}
  .sidebar-desktop{width:52px!important}
  .main-content{margin-left:52px!important}
}

@media(max-width:480px) and (orientation:portrait){
  .notes-split{flex-direction:column!important}
  .notes-ai-panel{display:none!important}
  .study-card-bottom{flex-direction:column!important;gap:8px}
  .study-card-know-row{flex-direction:row!important;gap:8px;width:100%}
  .study-card-know-row button{flex:1}
  .mcq-options{gap:8px!important}
  .mcq-option{padding:12px 14px!important;font-size:14px!important}
  .cards-grid{grid-template-columns:1fr!important}
  .quick-actions{grid-template-columns:1fr 1fr!important}
  .folder-file-row{flex-wrap:wrap!important}
  .folder-file-actions{width:100%!important;justify-content:flex-end!important;margin-top:4px}
  .view-split{flex-direction:column!important}
  .view-ai-panel{width:100%!important;max-height:260px;border-left:none!important;border-top:1px solid rgba(255,255,255,.1)}
  .modal-inner{border-radius:18px 18px 0 0!important;position:fixed!important;bottom:0!important;left:0!important;right:0!important;width:100%!important;max-width:100%!important}
  .notes-toolbar{flex-wrap:wrap!important;gap:6px!important}
  .notes-toolbar-right{flex-wrap:wrap!important;gap:6px!important}
}

@media(max-width:900px) and (orientation:landscape) and (max-height:480px){
  .bottom-nav{height:48px!important}
  .bottom-nav button{padding:2px 4px!important}
  .bottom-nav .nav-label{display:none!important}
  .study-mode-wrap{padding:6px 16px!important}
  .study-card-area{padding:8px 16px!important}
  .app-header-file{display:none!important}
  .notes-ai-panel{display:none!important}
  .view-ai-panel{width:280px!important;max-width:280px}
  .ad-banner-wrap{display:none!important}
  .page-with-ad{padding-bottom:0!important}
}

@supports(padding: max(0px)){
  .bottom-nav{
    padding-bottom: max(0px, env(safe-area-inset-bottom))!important;
    height: calc(56px + env(safe-area-inset-bottom))!important;
  }
  .page-with-ad{
    padding-bottom: calc(60px + env(safe-area-inset-bottom))!important;
  }
}

ins.adsbygoogle{max-height:46px!important;overflow:hidden!important}
ins.adsbygoogle iframe{max-height:46px!important}
button.folder-color-btn,button.color-swatch,button.no-min-h{aspect-ratio:1;flex-shrink:0;min-height:unset!important;min-width:unset!important;padding:0}

@keyframes bounce{0%,80%,100%{transform:scale(.8);opacity:.5}40%{transform:scale(1.1);opacity:1}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
@keyframes sg-pulse{0%,100%{box-shadow:0 0 0 0 rgba(74,124,89,.6)}70%{box-shadow:0 0 0 6px rgba(74,124,89,0)}}
@keyframes ppbar{0%,100%{transform:scaleY(.4);opacity:.5}50%{transform:scaleY(1);opacity:1}}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes twcursor{0%,100%{opacity:1}50%{opacity:0}}
`;

const FILE_STORE = new Map();

// ─── INDEXEDDB FILE PERSISTENCE ──────────────────────────────────────────────
const IDB_NAME = "classio_files";
const IDB_STORE = "files";

function openIDB() {
  return new Promise((res, rej) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = e => e.target.result.createObjectStore(IDB_STORE);
    req.onsuccess = e => res(e.target.result);
    req.onerror = () => rej(req.error);
  });
}

async function idbSave(id, file) {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).put(file, id);
    await new Promise((res, rej) => { tx.oncomplete = res; tx.onerror = rej; });
  } catch(e) { console.warn("IDB save failed", e); }
}

async function idbGet(id) {
  try {
    const db = await openIDB();
    return new Promise((res, rej) => {
      const req = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(id);
      req.onsuccess = () => res(req.result || null);
      req.onerror = () => res(null);
    });
  } catch(e) { return null; }
}

async function idbDelete(id) {
  try {
    const db = await openIDB();
    const tx = db.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).delete(id);
  } catch(e) {}
}

async function idbGetAll() {
  try {
    const db = await openIDB();
    return new Promise((res, rej) => {
      const result = {};
      const req = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).openCursor();
      req.onsuccess = e => {
        const cursor = e.target.result;
        if (cursor) { result[cursor.key] = cursor.value; cursor.continue(); }
        else res(result);
      };
      req.onerror = () => res({});
    });
  } catch(e) { return {}; }
}

function stripBlobs(flds) {
  return (flds || []).map(fo => ({
    id: fo.id || "",
    name: fo.name || "",
    color: fo.color || "#3D5A80",
    files: (fo.files || []).map(fi => ({
      id: fi.id || "",
      name: fi.name || "",
      type: fi.type || "",
      size: fi.size || 0,
      colorIndex: fi.colorIndex || 0,
      notes: fi.notes || "",
      studyCards: fi.studyCards || [],
      uploadedAt: fi.uploadedAt || "",
      linkedFileIds: fi.linkedFileIds || [],
    })),
  }));
}

// ─── STANDALONE AI ASSISTANT ─────────────────────────────────────────────────
function StandaloneAI({ onClose }) {
  const [msgs, setMsgs] = useState([]);
  const [inp, setInp] = useState("");
  const [loading, setLoading] = useState(false);
  const [attachedImage, setAttachedImage] = useState(null);
  const imgInputRef = useRef(null);
  const bottomRef = useRef(null);
  const theme = useTheme();

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);
  
  const attachImage = (f) => {
    if (!f) return;
    const r = new FileReader();
    r.onload = e => setAttachedImage({ base64: e.target.result, name: f.name });
    r.readAsDataURL(f);
  };

  const send = async () => {
    const text = inp.trim();
    if ((!text && !attachedImage) || loading) return;
    const content = text || "Analyze this image and explain or solve it.";
    const userMsg = { role:"user", content, image: attachedImage?.base64 };
    const newMsgs = [...msgs, userMsg];
    setMsgs(newMsgs); setInp(""); setLoading(true);
    const imgToSend = attachedImage?.base64 || null;
    setAttachedImage(null);
    try {
      const sys = "You are Classio AI, a smart study assistant. Help students with any question — solve problems, explain concepts, analyze images of questions or diagrams. Be clear and concise. No asterisks, no markdown.";
      const apiMsgs = newMsgs.map(m => ({ role:m.role, content:m.content }));
      const reply = imgToSend
        ? await callClaudeVision(sys, apiMsgs, imgToSend)
        : await callClaudeChat(sys, apiMsgs);
      setMsgs([...newMsgs, { role:"assistant", content: reply }]);
    } catch(e) { setMsgs([...newMsgs, { role:"assistant", content:"Error: " + e.message }]); }
    setLoading(false);
  };

  return (
    <div style={{ position:"fixed", inset:0, zIndex:3000, background:"rgba(26,23,20,.55)",
      backdropFilter:"blur(4px)", display:"flex", alignItems:"center", justifyContent:"center", padding:16 }}>
      <div style={{ background:theme.bg, borderRadius:22, width:"100%", maxWidth:680,
        height:"85vh", display:"flex", flexDirection:"column",
        boxShadow:"0 24px 80px rgba(0,0,0,.2)", border:`1px solid ${theme.border}`, position:"relative", zIndex:3001 }}>
        {/* Header */}
        <div style={{ display:"flex", alignItems:"center", gap:12, padding:"16px 20px",
          borderBottom:`1px solid ${theme.border}`, flexShrink:0 }}>
          <div style={{ width:40, height:40, borderRadius:12,
            background:"linear-gradient(135deg,#6366f1,#8b5cf6)",
            display:"flex", alignItems:"center", justifyContent:"center" }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="5" y="5" width="14" height="11" rx="2"/><path d="M9 10h.01M15 10h.01M9 13s1 1.5 3 1.5 3-1.5 3-1.5"/><path d="M12 16v2M8 20h8M12 5V3"/><circle cx="12" cy="3" r="1"/></svg>
          </div>
          <div style={{ flex:1 }}>
            <p style={{ margin:0, fontSize:16, fontWeight:700, color:theme.text }}>AI Assistant</p>
            <p style={{ margin:0, fontSize:12, color:theme.muted }}>Ask anything · Attach images · Replies in your language</p>
          </div>
          <button onClick={onClose} style={{ background:theme.surface, border:`1px solid ${theme.border}`,
            borderRadius:"50%", width:32, height:32, cursor:"pointer", fontSize:18,
            color:theme.muted, display:"flex", alignItems:"center", justifyContent:"center" }}>×</button>
        </div>
        {/* Messages */}
        <div style={{ flex:1, overflowY:"auto", padding:"16px 20px",
          display:"flex", flexDirection:"column", gap:12 }}>
          {msgs.length === 0 && (
            <div style={{ textAlign:"center", padding:"32px 16px" }}>
              <div style={{ width:64, height:64, borderRadius:20, background:"linear-gradient(135deg,#6366f1,#8b5cf6)", display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 14px" }}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="5" y="5" width="14" height="11" rx="2"/><path d="M9 10h.01M15 10h.01M9 13s1 1.5 3 1.5 3-1.5 3-1.5"/><path d="M12 16v2M8 20h8M12 5V3"/><circle cx="12" cy="3" r="1"/></svg>
              </div>
              <p style={{ fontSize:17, fontWeight:700, color:theme.text, marginBottom:8 }}>Classio AI</p>
              <p style={{ fontSize:13, color:theme.muted, lineHeight:1.7, maxWidth:320, margin:"0 auto 24px" }}>
                Ask me anything! Solve problems, explain concepts, or send a photo of a question.
              </p>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, maxWidth:360, margin:"0 auto", textAlign:"left" }}>
                {[["⬡","Photo of a question","Take a photo — I'll solve it"],
                  ["∑","Math & Science","Step-by-step solutions"],
                  ["✎","Explain concepts","Clear simple explanations"],
                  ["◎","Any language","Reply in your language"]].map(([ic,ti,de])=>(
                  <div key={ti} style={{ background:theme.surface, border:`1px solid ${theme.border}`,
                    borderRadius:12, padding:"12px 14px" }}>
                    <div style={{ marginBottom:6, color:theme.accent }}>{ic}</div>
                    <p style={{ margin:"0 0 2px", fontSize:12, fontWeight:700, color:theme.text }}>{ti}</p>
                    <p style={{ margin:0, fontSize:11, color:theme.muted, lineHeight:1.4 }}>{de}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
          {msgs.map((m,i)=>(
            <div key={i} style={{ display:"flex", justifyContent:m.role==="user"?"flex-end":"flex-start",
              alignItems:"flex-end", gap:8 }}>
              {m.role==="assistant" && (
                <div style={{ width:28,height:28,borderRadius:8,
                  background:"linear-gradient(135deg,#6366f1,#8b5cf6)",
                  display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0 }}>
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/></svg>
                </div>
              )}
              <div style={{ maxWidth:"80%", background:m.role==="user"?theme.accent:theme.surface,
                color:m.role==="user"?"#fff":theme.text, border:m.role==="user"?"none":`1px solid ${theme.border}`,
                borderRadius:16, padding:"12px 16px", fontSize:14, lineHeight:1.6 }}>
                {m.image && <img src={m.image} alt="attached" style={{ maxWidth:"100%", borderRadius:8, marginBottom:8 }} />}
                <Fmt text={m.content} />
              </div>
            </div>
          ))}
          {loading && (
            <div style={{ display:"flex", gap:8, alignItems:"center" }}>
              <div style={{ width:28,height:28,borderRadius:8, background:"linear-gradient(135deg,#6366f1,#8b5cf6)", display:"flex",alignItems:"center",justifyContent:"center" }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/></svg>
              </div>
              <p style={{ margin:0, fontSize:13, color:theme.muted }}>Thinking...</p>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        {/* Input Bar */}
        <div style={{ padding:"12px 16px", borderTop:`1px solid ${theme.border}`, flexShrink:0 }}>
          {attachedImage && (
            <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:8, background:theme.surface, padding:"4px 8px", borderRadius:8, border:`1px solid ${theme.border}` }}>
              <span style={{ fontSize:12, color:theme.text }}>📷 {attachedImage.name}</span>
              <button onClick={()=>setAttachedImage(null)} style={{ background:"none", border:"none", cursor:"pointer", color:theme.muted }}>×</button>
            </div>
          )}
          <div style={{ display:"flex", gap:8, alignItems:"center" }}>
            <input type="file" accept="image/*" ref={imgInputRef} style={{ display:"none" }} onChange={e=>attachImage(e.target.files[0])} />
            <button onClick={()=>imgInputRef.current?.click()} style={{ background:theme.surface, border:`1px solid ${theme.border}`, borderRadius:10, width:38, height:38, display:"flex", alignItems:"center", justifyContent:"center", cursor:"pointer", color:theme.muted }}>📷</button>
            <input value={inp} onChange={e=>setInp(e.target.value)} onKeyDown={e=>e.key==="Enter"&&send()} placeholder="Ask Classio AI..." style={{ flex:1, background:theme.surface, border:`1px solid ${theme.border}`, borderRadius:10, padding:"8px 12px", fontSize:14, color:theme.text, outline:"none" }} />
            <button onClick={send} disabled={loading} style={{ background:theme.accent, color:"#fff", border:"none", borderRadius:10, padding:"0 16px", height:38, fontWeight:600, cursor:"pointer" }}>Send</button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── MAIN CLASS COMPONENT / EXPORT ───────────────────────────────────────────
export default function App() {
  const theme = useTheme();

  return (
    <div style={{ background: theme.bg, color: theme.text, minHeight: "100vh", fontFamily: "'DM Sans', sans-serif" }}>
      <style>{GS}</style>
      <StandaloneAI onClose={() => {}} />
    </div>
  );
}