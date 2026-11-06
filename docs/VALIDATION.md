# Executed validation

Date: 6 November 2026. These results apply to this local development branch,
not a production deployment or an independently restored cloud task.

## Passed

- Fresh locked install: `npm ci --cache /workspace/.cache/npm --no-audit --no-fund`.
  Node 24.22.1 / npm 11.9.0. All 1,996 packages installed; deprecation warnings remain.
- `npm run build`: optimized production build completed with lint warnings.
  The legacy app still has lint warnings; this is not a zero-warning build.
- `CI=true npm test -- --watchAll=false --runInBand`: **33 tests, 5 suites passed**.
  Covers eligibility UI, document tools, safe content, YouTube normalization,
  request cancellation/timeouts and partial file-upload failure behavior.
- `npm run test:server`: **29 tests passed**. Provider HTTP responses and database
  services are mocked. Covers Groq/Gemini REST formats, preference/consent,
  bounded fallback, rate limits, invalid keys/models, known-credential redaction,
  signed cookies, account scoping, course validation, idempotency and save failure.
  This does not establish real provider generation or real Firebase authorization.
- `PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_server` in the speech repo:
  **6 tests passed**, with Python 3.11.9. Synthesis and model-download streams are
  test doubles where needed. Covers the HTTP WAV contract, validation, exception
  privacy, checksum rejection and recovery from incomplete downloads.
- Local speech HTTP `/health` and `/voices` returned the expected successful responses. A real `/generate-podcast` request returned a safe HTTP 503 because model access is blocked; no real WAV was generated.
- JSX parsing of the retained legacy speech-repo `App.js` passed. This file has no
  independent frontend build manifest and is not served by the Python application.
- `npm run test:browser` against the real local frontend/API in system Chromium:
  signed eligibility/terms, year-long HttpOnly cookie, refresh, blocked age decision,
  ignoring a forged legacy storage flag, navigation/drawer/Escape, Assistant open,
  theme persistence and page overflow checked at **320×740, 390×844, 844×390,
  768×1024, 1024×768, 1366×900 and 1920×1080**.
- Browser workflows: actual named-folder YouTube URL save/reopen; actual text-file
  upload/view; note editing retained through navigation; real local unconfigured-AI
  error/retry without fake responses or duplicate messages; chat draft after refresh.
- Browser course rendering, progress, editing and reopening passed with an explicit
  intercepted AI response fixture. This is a frontend workflow check, not a claim
  that a real AI provider generated or cloud-saved that course.
- Production build scanned across **99 generated files**, including source maps,
  with temporary private Groq/Gemini environment sentinel values: neither appeared.
- Targeted available-history scan: **265 Classio blobs and 18 speech-repo blobs**
  inspected for Groq/Anthropic literal key prefixes and PEM private keys. No matches.
  This does not inspect past hosted bundles, deployment logs, encoded secrets or
  every possible credential format. Firebase web configuration is public project
  identification, separate from the server-only AI/Admin credentials.
- No Signum references found in the searched app/legacy source and public assets.
  No production-page claim is made from this source search.
- `git diff --check` passed for both repositories.

## Blocked or unrun

- **Firestore/Storage emulator tests did not execute.** The CLI attempted to fetch
  the emulator JAR from storage.googleapis.com; the proxy returned HTTP 403.
  `npm run test:rules` contains six real SDK/rules tests for same-UID two-device
  original recall, tenant isolation, immutable originals and protected records.
  Their assertions have not been reported as passed.
- **No live Groq or Gemini request.** No provider credentials are present here.
  Current official model/pricing/terms documentation is blocked by network policy.
- **No production Google login, account linking or logout.** Real identity-provider
  configuration, domain authorization and credentials are unavailable for this check.
- **No real cross-device cloud file recall or migration against Firebase.** This
  needs the exact Storage bucket, reviewed deployed rules, project access and tests.
- **No real Piper audio generation.** Hugging Face model access is blocked. The
  pinned voice catalog/release must be verified with every enabled voice before release.
- No live YouTube transcript retrieval, playback of restricted/private videos or
  physical iPhone/iPad Safari testing. Browser checks cover URL/UI handling only.
- No complete audit of every legacy feature, dark-mode control, overlay, dependency
  vulnerability, device combination or WCAG 2.2 AA criterion.
- No production deployment or production credential/rule changes performed.

## Reproduce locally

```sh
npm ci
npm run test:server
CI=true npm test -- --watchAll=false --runInBand
npm run build
# In another terminal, with a local Chromium and configured development settings:
npm start
npm run test:browser
# With Java 21 and emulator downloads permitted:
npm run test:rules
```

The browser script defaults to `/usr/bin/chromium`; set `CLASSIO_BROWSER_PATH`
for another installed Chromium. It writes ignored screenshots into `test-results/`.
Do not publish or merge this branch as a fully validated production upgrade until
its required external configuration and integration checks are complete.
