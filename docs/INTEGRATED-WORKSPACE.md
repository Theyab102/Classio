# Classio integrated workspace update

For the later verification, village-character, Assistant-action and dependency
changes, see [VERIFICATION-AND-VILLAGERS.md](VERIFICATION-AND-VILLAGERS.md).

6 November 2026. App branch: `feat/integrated-study-and-podcasts`.
Speech branch: `feat/two-host-audio-and-captions` in `Theyab102/classio-tts-server`.
These branches include the preceding critical-functionality and calm-palette work.
Neither production branch nor production data has been changed by this delivery.

## What changed

- One document editor for manual and AI notes, with a compact formatting header,
  collapsible generation settings, local font controls, tables, images and versions.
  **Generate AI notes** reads the actual file, accessible transcript or supplied
  source text. It appends without replacing existing writing. Organization keeps
  a previous version. A missing transcript is not replaced by invented notes.
- A document chat beside the editor, with the layout of the supplied reference:
  slim navigation, centered document, right conversation/tools panel. Smaller
  screens open chat in a dialog. Quiz, flashcard and podcast shortcuts open the
  existing functional tools. Messages and per-file history use workspace sync.
- Safe shared Markdown rendering now includes real tables, emphasis, code,
  KaTeX mathematics and the existing Mermaid diagrams. Model-generated raw HTML
  is escaped; Markdown images do not make remote tracking requests.
- The requested `#2b6777 / #c8d8e4 / #ffffff / #f2f2f2 / #52ab98` palette is retained.
  The dark document view uses the reference's quiet charcoal surfaces and teal actions.
- Original console-style character artwork with rounded proportions, classic
  oval eyes, sculpted noses, hair, expressions, glasses and accessories. Live
  controls include feature sizes/positions and custom colors. Existing appearance
  settings and account persistence are preserved. No Nintendo assets or Turbo AI
  branding were copied. These are original Classio characters, not licensed Miis.
- English/Arabic core interface and RTL support now cover the upgraded course,
  recording, podcast and chat workflows. Settings also select an AI response
  language. Some older games, groups, help text and policy prose remain English.
- Portable request timeouts, native dialog fallback/focus handling, safe areas,
  and visible-viewport sizing improve small screens and keyboard-constrained layouts.
  This does not establish physical iOS Safari compatibility.

## AI routing, access and persistence

The shared server REST adapters remain in `server/ai-providers.js`; browser code
only calls Classio's backend. Auto prefers configured Groq. An alternative provider
gets at most one attempt, only with the user's fallback permission and a transient
failure/rate limit. Invalid configuration does not trigger endless retries. Returned
metadata identifies the provider. Transcription deliberately uses Groq's speech API.

Gemini requires `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_ENABLED=true` and
`GEMINI_FREE_TIER_VERIFIED=true`. It additionally requires a trusted adult-session
decision from Classio's signed access/account record, never a browser flag. This
gate does **not** prove a mixed-age deployment is eligible under Google's current
terms. The operator must verify current audience/data-use terms, free-model quotas
and project billing before enabling it. No paid service is enabled by this code.

`/api/agent` offers the validated account-owned `create_course` workflow, including
database confirmation and request IDs. The Assistant opens that workflow. It does
not autonomously select arbitrary tools or execute commands/database queries.

Google authentication, signed persistent age decisions, private Storage originals,
account-scoped notes, conversations, profiles, themes and progress use the existing
Firebase project. A returning previously signed-in account is given time to restore
its server verification before showing the age screen. Network failures offer retry.
Age screening is self-declaration; it is not identity verification.

Original-file recall is by the **same Firebase Google account UID**, not matching
email text. Uploads must finish before they are available on the other device.
Originals are immutable, SHA-256 verified and limited to 20 MiB. Old browser-only
originals must be recovered from that original browser or uploaded again. Legacy
arrays are retained during the additive migration. See `UPGRADE.md` for the schema.
No production migration or two-device Firebase recall was executed here.

## Podcast and microphone workflow

`PodcastStudio.js` sends actual study material to the shared AI endpoint for a
validated one/two-host English dialogue. Topic, length target, difficulty, style
and distinct voice choices are editable. The six installed Piper voices are English;
other audio languages are not advertised as supported. Lengths are targets.

`/api/podcast` validates access and quotas and issues a ten-minute HMAC ticket bound
to a pseudonymous owner and exact script/voice fingerprint. The Python service
validates that ticket for job creation, status and audio. Durable SQLite jobs are
idempotent; polling/reopening reuses the same completed job. One service process
and a persistent private volume are required for restart recovery.

The server synthesizes complete phrases, trims silence, applies fades and peak/RMS
limits, streams assembly to disk, and encodes AAC/M4A with a WAV fallback. Captions
come from actual phrase sample durations. They highlight the phrase and speaker,
not individual words. The player uses the audio element's current time for seeking,
speed, captions and speaker animation; bars use the generated audio's RMS envelope.
This is synchronized speaking animation, not lip sync. iOS users use device volume
if the browser rejects programmatic volume changes.

Completed audio is cached in IndexedDB; signed-in users can sync it to private
Firebase Storage. The UI distinguishes a local cache, uploaded bytes, and workspace
metadata sync. Scripts survive a synthesis error. Successful audio can replay
without generation. Download is available; server cache expires after 24 hours by
default, while a saved local/account copy persists until cleared by the user/operator.

`VoiceCapture.js` records using MediaRecorder with M4A/WebM support, stops at two
minutes, and offers preview/download/keep-original before any transcription. It
only sends audio after **Transcribe with Groq**. Unfinished recordings/drafts use
account-scoped or guest-session-scoped local IndexedDB, with a leaving warning.
Legacy browser speech/dictation remains available as a separate collapsed tool.

## Privacy and licensing changes

DM Sans, Fraunces, Noto Sans Arabic and KaTeX fonts are served from Classio. Actual
WOFF2 files, font hashes and licenses are included under `public/fonts` and
`public/licenses`. No Google Fonts imports or font CDN calls are needed. Mermaid,
Mammoth, Chart.js and spreadsheet reading now use local, lazy-loaded build chunks.
Legacy `.xls` needs conversion to `.xlsx`; originals remain available to download.

There is no session replay or keystroke analytics SDK in the searched source. Normal
editor input handlers and debounced account autosave still implement editing/saving.
Typing a note or chat draft does not itself send an AI request. Microphone capture
requires a deliberate action. YouTube previews and the optional Google document
viewer load only after explicit choice; these external services then receive an IP
address. Selected AI content goes to the chosen provider. Firebase authentication
and account storage also contact Google; local fonts do not eliminate those services.

Vercel headers restrict font/style sources, referrers, framing and unused device
permissions. The same-origin Firebase auth proxy remains compatible with framing.
Policies describe these actual flows. They still need the real operator/contact
fields and jurisdiction-specific legal review. No code change can guarantee immunity
from lawsuits or that third-party artwork/voice licenses cover every planned use.
Review Piper model cards and their underlying dataset licenses before release.

## Configure and deploy the two projects

1. Review both development branches. Keep the existing Firebase project and data.
   Use Node 22/24 and `npm ci` for Classio; Python 3.11 and the locked requirements
   for the speech service. See that repository's README for persistent storage,
   checksummed models and single-worker startup.
2. Configure Vercel server-only `GROQ_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_JSON`,
   `AI_RATE_LIMIT_SECRET` and preferably separate `CLASSIO_SESSION_SECRET` (32+
   random characters). Reuse valid secure values. Configure Gemini only after the
   eligibility/free-tier review above. Create a replacement key securely through
   [AI Studio](https://aistudio.google.com/apikey) if needed. Never paste it in chat
   or use a frontend-prefixed variable. Revoke the credential pasted in this chat.
3. Set the same private random `CLASSIO_TTS_SIGNING_SECRET` on Vercel and the Python
   service. Set the public build variable `REACT_APP_CLASSIO_PODCAST_URL` to the new
   job server. Preserve `REACT_APP_CLASSIO_TTS_URL` for the existing transcript
   service if it is separate. These URLs contain no secret. Add production and
   authorized preview origins to the Python service's `CLASSIO_ALLOWED_ORIGINS`.
4. For Safari-friendly Firebase redirect helpers, the Vercel rewrite proxies
   `/__/auth/*` to `classio-4378f.firebaseapp.com`. On the production app set
   `REACT_APP_FIREBASE_AUTH_DOMAIN=classio-eta.vercel.app`; add that domain in Firebase
   Authentication and allow `https://classio-eta.vercel.app/__/auth/handler` in the
   existing Google OAuth client. Use matching values for a final custom domain.
   Verify against the actual Firebase console settings before deployment. A console
   URL alone does not grant configuration access.
5. Set the exact public `REACT_APP_FIREBASE_STORAGE_BUCKET`, configure private
   Storage/CORS and review Firebase's current plan/quota requirements. No bucket,
   paid plan or billing setting was provisioned. **Merge** the supplied workspace
   rules with existing production rules, including working study-group rules.
   The supplied owner-workspace template does not define the legacy group schema;
   blindly replacing all production rules could disable those groups. Run staging
   SDK/emulator tests before applying rules. Do not erase records or legacy arrays.
6. Fill the public operator/contact variables from `.env.example`. Rebuild Vercel
   after changing build variables; redeploy after server settings. Review any Render
   disk/compute cost before applying the existing deployment descriptor.
7. Test real Groq and Gemini separately, allowed fallback, Google popup/redirect,
   logout/refresh, same-account recall on phone/iPad, real two-voice synthesis and
   physical Safari. Network/credential restrictions prevented those checks here.

## Reproduce local verification

```sh
npm ci
CI=true npm test -- --watchAll=false --runInBand
npm run test:server
npm run build
```

For internal Chromium QA, start `node tests/production-preview.cjs` after building.
In the speech repository start `python tests/fixture_server.py`, then run
`npm run test:integrated` in Classio. The fixture is explicitly synthetic and must
never be deployed; it tests real job authorization/assembly/player behavior, not
voice naturalness or live AI. `node tests/document-exports.cjs` checks real PPTX
export and creates Word/XLSX fixtures. `npm run test:browser` and `npm run test:ui`
exercise the earlier authentication/navigation/palette workflows. Run suites
sequentially when they share port 3000. The test server is loopback-only and is not
a public preview.

Current executed results and blockers are recorded in `INTEGRATED-VALIDATION.md`.
The historical `UPGRADE.md`, `VALIDATION.md` and `CALM-UI.md` describe earlier passes.
