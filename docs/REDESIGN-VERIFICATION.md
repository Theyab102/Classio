# Classio redesign verification

Validated locally on 2026-11-07. No user content or account records were reset.

## Automated results

| Check | Result |
| --- | --- |
| Production build, including CRA ESLint checks | Passed; legacy warnings remain in `src/App.js`, with no build errors or warnings in the new components |
| React/component regressions | 15 suites, 51 tests passed |
| Node API/provider regressions | 21 tests passed |
| Playwright full study flows | Passed at 1440 × 1000 and 390 × 844 |
| Additional responsive checks | Passed at 320, 768 and 1024 pixels in light and dark themes, with reduced motion |
| Browser runtime errors in tested flows | None |
| Frontend secret exclusion | A deliberately supplied `REACT_APP_GROQ_KEY` test sentinel was excluded from every built artifact, including source maps |
| Source credential-pattern scan | Passed; no provider credentials/private keys found. Existing public Firebase client configuration is distinct from provider secrets |
| Whitespace/conflict checks | `git diff --check` passed |

Run `npm run build`, `CI=true npm test -- --watchAll=false --runInBand`, `npm run test:server` and `npm run test:browser` to repeat these checks. The browser script prints its screenshot directory. Set `CLASSIO_BROWSER_OUTPUT` for a stable directory and `CLASSIO_CHROMIUM_PATH` if Chromium is installed separately.

## Browser coverage

- The four restored home actions are visible and invoke their existing handlers. Real user materials are searchable, sortable and movable into a newly created folder. File deletion is tested both with Cancel and explicit confirmation.
- Notes support writing, formatting, actual KaTeX equations and tables. A failed AI request displays an error. Generated drafts do not replace writing until applied. Named versions remain available.
- The visible composer opens the shared assistant. The request contains the selected material. Image attachment requests pass the actual backend schema. History survives closing, reopening and navigating. The desktop panel reserves space; smaller layouts keep the composer accessible.
- Lessons retain knowledge-check completion and the active lesson. Quizzes support hints, checked answers, navigation, results and explanations. Flashcards support generation, editing, flipping, study grading and saved review categories.
- A speech-service failure saves the podcast transcript and displays no pretend player. Preparing audio again does not regenerate the script. Actual WAV bytes are decoded and played with duration, seek, speed, volume, speaker transcripts and playback position restored after navigation.
- The avatar editor displays actual hairstyle/outfit previews. Save commits the name/look; Cancel discards edits. Separate component tests verify that changing accounts cannot inherit another account's character or conversation.
- Real generated PDF and PPTX fixture files upload through the UI. Both source viewers support search, navigation, zoom and both themes. PDF pen annotations, page and zoom survive navigating away and back. The native PDF aspect ratio remains intact.
- Presentation generation and design selection remain functional, including editable tables. Study games remain reachable. An unavailable video transcript does not trigger invented title-based notes.
- A refreshed guest session starts without the previous guest's materials while keeping a valid age/terms decision. Account restoration regressions cover unavailable Firebase, blocked local file storage and an unauthorized Google-login hostname.

AI and speech **transport responses are explicit test fixtures**. The speech fixture is playable PCM audio, not a transcript disguised as audio. These tests validate application behavior; they do not establish live Groq, Gemini, YouTube or Piper availability. Live Google login is also not claimed from mocked account tests.

## Production configuration and remaining checks

The owner confirmed that both credentials exposed in screenshots were replaced. The latest user-provided response from the old deployment was:

```json
{"aiConfigured":false,"providers":[],"missing":["GROQ_API_KEY","FIREBASE_SERVICE_ACCOUNT_JSON","AI_RATE_LIMIT_SECRET"],"checked":"configuration only; live provider access is not tested"}
```

The updated backend accepts the confirmed `Groq_Key` and `Gemini_Key` aliases as server-only credentials. It discovers supported models, performs bounded compatible fallback, and applies a brief cooldown after rejected credentials or rate limits. Status output contains names and booleans, never secret values.

Default `instance` quotas no longer require a Firebase service account. They reset with function instances and are not a shared deployment-wide budget. Optional `firestore` mode retains shared transactional quotas and requires both the admin credential and rate-limit secret. Gemini remains disabled until `GEMINI_ENABLED=true` is explicitly configured consistently with the app's audience and provider authorization; a key by itself does not enable it.

The executor's enforced cloud network rejects the Production URL with proxy HTTP 403. No Vercel identity, provider secrets or Firebase deployment credentials are available here. Consequently these checks remain external:

1. Confirm Vercel builds the pushed main commit successfully and `/api/status` reports the expected recognized setting names/providers.
2. Send actual guest text, image, notes/course/quiz/card and transcription requests from Production, and verify real provider responses and any configured fallback.
3. Test the existing Piper speech Space from Production, including its actual voice inventory and audio generation after waking.
4. Review and deploy the additive owner-only `users/{uid}/preferences/assistant` Firestore rule to enable cross-device conversation sync. Browser-only history remains available with a truthful sync notice if this path is denied. Do not replace unrelated live project rules without reviewing them.
5. Confirm Google login and cross-device restore with a real account on the deployed hostname.

Original uploads and generated podcast audio remain device-local, with transcripts/metadata kept in the workspace. Complex PowerPoint effects and scanned-image PDF text search are limited by the existing native viewers. Browser speech recognition and read-aloud are shown only where supported; no live voice-conversation feature is advertised.
