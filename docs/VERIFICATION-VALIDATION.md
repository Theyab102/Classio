# Verification/village pass: executed results

6 November 2026. These apply to local development branches, not production.

## Passed

- Fresh locked npm install after dependency overrides; production build succeeded
  with legacy lint/deprecation warnings. Error-level ESLint passed.
- **74 frontend tests, 16 suites**: includes slow/failed access, malformed responses,
  ignored cookies, bounded restoration, online recovery, fallback UUIDs, village
  customization, actual Assistant proposal/confirmation UI, failed saves,
  conversation isolation during background actions, quiz progress and bounded
  account linking. Firebase/provider responses are doubles where required.
- **47 Node backend tests**: signed eligibility, account scope, provider routing,
  quota/error handling, course validation and ten new study-action tests. They
  verify ownership, review-only planning, separate output files, all four output
  types, explicit confirmation, retries without duplicate generation/quota,
  destructive proposal rejection, changed sources and transactional save failures.
- **18 Python tests** on **Python 3.11.17**, after an exact hash-locked install:
  HTTP validation, signed/private jobs, ownership/fingerprint/expiry, SQLite
  recovery/replay, PCM boundaries/limits and real FFmpeg encoding. Speech synthesis
  is a fixture where needed. Bundled eSpeak phonemization also passed.
- Normal patched development-server startup and real local access requests passed.
  An untrusted Host/Origin was rejected with HTTP 403. Normal Python process health,
  voices and capabilities returned valid HTTP 200; an absent signing secret was
  correctly reported as unconfigured, not successful speech readiness.
- `scripts/verify-deployment.cjs` against the real local API passed new-session,
  date screening, terms, signed-cookie retention and returning accepted access.

## Browser workflows against the production build

- `tests/background-verification.cjs`: welcome visible during a held background
  request; no checking screen or premature birth-date form; actual cookie flow;
  refresh/offline failure does not discard verification; automatic connectivity
  recovery; discarded-cookie recovery; six village species and five phone/tablet
  viewport fits. Dark entry and village editor Axe samples had zero violations.
- `tests/local-documents.cjs` and `tests/document-exports.cjs`: actual local DOCX,
  XLSX and PPTX rendering, real PPTX export with image/table, constrained visual
  viewport and older-dialog focus/Escape behavior.
- `tests/integrated-workspace.cjs`: actual text source, notes/chat, real Markdown
  table and font controls; no AI submission on typing; explicit AI output fixture;
  real signed local podcast job/SQLite/assembly/AAC pipeline using synthetic speech;
  actual HTMLAudio pause/seek/speed/volume/caption/replay; actual MediaRecorder
  capture with a simulated microphone and deliberate transcription fixture.
  Six viewport sizes passed. All observed font/application scripts stayed local.
  The five sampled dashboard/notes/character/podcast Axe states had zero violations.
- Existing `test:browser` and `test:ui` suites passed seven/six size sets covering
  persistent signed eligibility, forged-storage rejection, navigation/drawer,
  Assistant, YouTube named-folder references, course/notes fixtures, themes,
  English/Arabic RTL, reduced motion and live avatar customization/browser recall.

This is an automated sample, not whole-app WCAG certification. Synthetic audio and
intercepted AI responses do not prove live provider generation, natural voices,
production authentication, real cloud saves or physical Safari behavior.

## Security checks

- npm audit after targeted compatible overrides: **30 findings — 0 critical,
  0 high, 12 moderate, 18 low**. The seven prior high findings were removed.
  Deprecated CRA/build dependencies remain; this is not a clean npm audit.
- `pip-audit` after the speech dependency update and fresh hashed install reported
  **no known vulnerabilities** in the 33 installed project/test dependencies.
  Interpreter/operating-system security and unreported vulnerabilities are outside
  that dependency result. The tested interpreter is Python 3.11.17.
- Private-key-pattern/source/build/archive scans and build-only server credential
  sentinels were checked before delivery. Firebase's public project config and
  explicit synthetic test tokens are separate from real private credentials.
  Scan summaries and exact artifact hashes are included in the review package.
- No production secret values, original personal uploads, generated private audio,
  databases, model weights or real dotenv files are included in the source ZIPs.
  Local interface fonts and their license/checksum manifest remain bundled.
- `git diff --check` passed. No production data, rules or billing changes occurred.

## Not verified / still external

No securely bound Groq, Gemini, Firebase Admin/Storage or deployment identity was
available. The environment's custom-domain access remained inactive. The live
Vercel failure's exact cause, real provider responses/current Gemini eligibility,
Google OAuth, account verification across devices and actual cloud-file recall
therefore remain unverified. A Firebase console link alone grants no access.

Hugging Face/model downloads, WebKit and Firebase emulator downloads were blocked
in the earlier attempts. No policy change unblocked them, so those failed attempts
were not relabeled as passes. Real Piper quality/licensing, live YouTube captions
and physical iPhone/iPad Safari still require checks. Legacy localization, whole-app
accessibility/visual review and operator/legal details remain unfinished.

Separate review branches are used; see the handoff for commit and push status.
Production is not merged or deployed. GitHub API access for creating PRs remains blocked; prepared descriptions
and comparison links accompany the files. The saved environment draft is separate
from published settings and active credentials.
