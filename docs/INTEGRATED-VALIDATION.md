# Integrated-workspace verification

Historical integrated-pass results. The later results are in
[VERIFICATION-VALIDATION.md](VERIFICATION-VALIDATION.md).

Executed locally on 6 November 2026. These are development-branch results, not
production deployment or physical-device results. See `INTEGRATED-WORKSPACE.md`
for setup and the implemented boundaries.

## Passed

| Check | Executed result |
| --- | --- |
| Locked fresh setup | `npm ci`; Python 3.11.9 requirements and `uv pip check`; bundled eSpeak phonemization succeeded |
| Frontend tests | 56 tests in 11 suites; `CI=true npm test -- --watchAll=false --runInBand` |
| Backend tests | 35 tests; `npm run test:server` |
| Speech-server tests | 18 tests; `python -m unittest -v test_server test_podcasts` |
| Source lint | `npx eslint src api server scripts/dev.cjs --quiet` passed; warning-level legacy issues remain |
| Production build | `npm run build` succeeded, with lint/deprecation warnings |
| Ordinary development startup | `npm start`; frontend HTTP 200, real local `/api/study` metadata HTTP 200 with both providers correctly unconfigured |
| Normal Python startup | `/health`, `/voices`, `/podcasts/capabilities` HTTP 200; missing signing secret correctly reported as unconfigured |
| Document exports/viewers | Actual PPTX export with image/table; actual local Chromium rendering of DOCX, XLSX and PPTX fixtures using local modules |
| Overlay constraints | Simulated reduced visible viewport; fallback dialog Tab trap, Escape and focus restoration |
| Security scans | No tested private-key patterns or real dotenv files in the scanned source; no server-key build sentinels in 246 production files including source maps |
| Git whitespace | `git diff --check` passed in both repositories |

Provider and database unit tests use explicit doubles. They verify REST payloads,
validation, bounded consented fallback, rate limits, rejected keys/models, access,
user scoping, idempotency, failed saves and known-credential response redaction;
they do not prove live generation or real Firebase authorization.

Python tests use explicit speech/model doubles where needed but exercise real
PCM conversion/assembly, SQLite jobs, ticket ownership/fingerprint/expiry,
recovery/replay, silent/failed segment handling and actual FFmpeg AAC encoding.
Catalog entries for the six medium voices and the high overrides were checked
against the installed pinned Piper catalog. No real voice-quality result is claimed.

## Chromium workflows

`tests/integrated-workspace.cjs` ran against the actual production build/local API
and the explicit local `tests/fixture_server.py` speech entry point:

- Upload/read a text source, normal note editing and source-grounded chat requests.
  Typing a note or unsent chat draft produces zero AI calls. Add-to-notes and
  generation preserve original writing in the same editor. Local font controls
  work. Assistant tables render as actual table elements.
- Real signed local podcast tickets, private job creation/status/download, real
  audio-element playback, pause, seeking, phrase captions, host activity, speed,
  volume, CC toggle, and replay without requesting another generation.
- Real MediaRecorder capture using Chromium's simulated hardware microphone;
  no background transcription; explicit Groq transcription response fixture and
  voice-note creation. This is not a real Groq transcription check.
- Document chat/dialog fit and no horizontal page overflow at 320×740, 390×844,
  844×390, 768×1024, 1024×768 and 1180×820.
- All observed application script/font requests stayed on Classio. No page errors.
- Axe WCAG 2 A/AA, 2.1 AA and 2.2 AA automated tags reported **zero violations**
  on the sampled dashboard, notes light, notes dark, character editor and podcast
  player states. This is an automated sample, not a whole-app WCAG certification.

The voice fixture generates sine waves; the AI-success requests are deliberately
intercepted fixtures. They establish player/request behavior, not natural speech,
factual AI quality, live credentials, cloud saves or cross-device Firebase recall.

`tests/browser-smoke.cjs` passed signed age/terms cookies, refresh, forged storage
rejection, blocked-age refresh, mobile drawer/Escape, Assistant navigation, theme
refresh and overflow at seven sizes (320×740 through 1920×1080). It also checked
real missing-configuration error/retry, chat recall, named-folder YouTube saving,
course error recovery, file/notes navigation, plus fixture-based course editing,
progress and reopening.

`tests/calm-ui-browser.cjs` passed the palette, one Assistant navigation entry,
same-editor AI notes, preserved writing, fixture-backed transcript notes,
live avatar customization, browser recall, English/Arabic Settings/Assistant,
RTL/LTR and response-language persistence, reduced motion and all avatar tabs
at six sizes (320×740 through 1366×900).

## Dependency and credential findings

`npm audit --omit=dev --json` completed and returned its expected nonzero finding
status: **36 findings: 7 high, 12 moderate, 17 low, 0 critical**. This fell from
48 before compatible patches to gRPC, image parsing and build helpers. The remaining
high-severity paths are the CRA/SVGO/PostCSS/webpack development/build chain.
`react-scripts` is declared under dependencies, so `--omit=dev` includes that
toolchain. Do not call the dependency audit clean or expose the development server
publicly. A separately reviewed CRA/build-tool migration remains necessary.

Targeted available-history scans checked 223 app commits and 12 speech commits
for Groq/OpenAI-style private tokens, the pasted-key format and PEM private keys;
no matching paths were found. Source/build scans check those selected patterns and
build-only server-variable sentinels, not every possible encoded/unknown secret.
Firebase's public web project key is intentionally public configuration, separate
from private AI/Admin/signing credentials. No deployment logs or historical hosted
bundles were available. Revoke the Gemini credential pasted in this chat; removing
it from a conversation or source would not revoke it.

No active Google Fonts imports or session-replay/keystroke-analytics SDK references
were found in the searched source. One sanitizer test contains a deliberate Google
Fonts URL as rejected input. No Signum references were found in the searched app
and legacy source. Ordinary editor events, autosave and explicit service actions
remain functional; this does not mean the app never contacts Firebase or Google.

## Blocked or unrun

- Live Groq/Gemini generation and current model/free-tier/audience documentation:
  no secure bound credentials and restricted external domains. Both provider
  adapters are implemented but neither was verified against its live API here.
- Production Google popup/redirect/logout, actual Firebase rules/migration and
  phone/iPad account-file recall. Project access, exact Storage bucket/configuration
  and reviewed deployed rules remain required.
- Firestore/Storage emulator tests: the attempt was blocked while downloading the
  emulator JAR from `storage.googleapis.com`; assertions did not execute.
- Real Piper model downloads, naturalness/pronunciation and real spoken captions:
  Hugging Face access was blocked. No live two-voice WAV was synthesized in this task.
- Playwright WebKit installation returned network-policy HTTP 403. Physical iPhone,
  iPad and Safari audio/keyboard/OAuth behavior have not been tested.
- Live/restricted/private YouTube behavior and live caption retrieval. URL/storage
  workflows and explicit transcript fallback fixtures were tested.
- Whole-app localization, every legacy dark-mode/overlay/accessibility control,
  autonomous model-selected tools beyond the explicit course workflow, and legal
  compliance. Operator/contact details and voice/license review remain outstanding.
- No production deployment, merge, database/rule replacement, secret rotation or
  billing change was performed. GitHub branches and downloadable sources are the
  review deliverables; API access to create PRs is blocked in this environment.

The saved environment draft includes tested install/start instructions, required
domains and variable/secret names. Saving that draft did not activate network
access, bind credentials or publish an environment snapshot.
