# Classio redesign verification

Validated locally on 2026-11-07. No user content or account records were reset.

## Automated results

| Check | Result |
| --- | --- |
| Production build, including CRA ESLint checks | Passed; legacy warnings remain in `src/App.js`, with no build errors or warnings in the new components |
| React/component regressions | 19 suites, 65 tests passed |
| Node API/provider regressions | 25 tests passed |
| Playwright full study flows | Passed at 1440 × 1000 and 390 × 844 |
| Additional responsive checks | Passed at 320, 768 and 1024 pixels in light and dark themes, with reduced motion |
| Browser runtime errors in tested flows | None |
| Frontend secret exclusion | A deliberately supplied `REACT_APP_GROQ_KEY` test sentinel was excluded from every built artifact, including source maps |
| Source credential-pattern scan | Passed; no provider credentials/private keys found. Existing public Firebase client configuration is distinct from provider secrets |
| Whitespace/conflict checks | `git diff --check` passed |

Run `npm run build`, `CI=true npm test -- --watchAll=false --runInBand`, `npm run test:server` and `npm run test:browser` to repeat these checks. The browser script prints its screenshot directory. Set `CLASSIO_BROWSER_OUTPUT` for a stable directory and `CLASSIO_CHROMIUM_PATH` if Chromium is installed separately.

## Browser coverage

- The four restored home actions are visible and invoke their existing handlers. Real user materials are searchable, sortable and movable into a newly created folder. File deletion is tested both with Cancel and explicit confirmation.
- Notes support writing, formatting, actual KaTeX equations and tables. A failed AI request displays an error. Generation fills an empty editor immediately and appends beneath existing writing without a review sidebar; a single Undo restores the prior content. Named versions remain available. Grouped image suggestions become actual source or licensed figures beside the related paragraph; unavailable images do not leave raw markers. Public webpage images use the same placement rules. An existing-note repair action removes old image placeholders while keeping writing, formatting and existing figures.
- The Ask Classio dock expands the right-side assistant; minimizing it preserves history and the draft. The request contains the selected material. Image attachment requests pass the actual backend schema. History survives closing, reopening and navigating. The desktop panel reserves space; smaller layouts keep the composer accessible. A short greeting sends a real small request without the current document. Histories and material excerpts are bounded, and provider rate-limit codes and retry intervals are retained.
- Guided lessons are an optional Notes mode, with explanations, worked examples and helpful questions. They retain knowledge-check completion and the active lesson; the duplicate Lessons navigation entry is removed. Quizzes support hints, checked answers, navigation, results and explanations. Flashcards support generation, editing, flipping, study grading and saved review categories.
- A speech-service failure saves the podcast transcript and displays no pretend player. Preparing audio again does not regenerate the script. Actual WAV bytes are decoded and played with duration, seek, speed, volume, speaker transcripts and playback position restored after navigation. Leading speaker labels are stripped before synthesis. A synthetic browser microphone produces a real recording, which passes the audio request schema; the transcribed question, grounded reply, real WAV answer and episode resumption are exercised with explicit transport fixtures.
- The avatar editor renders original Three.js meshes for full-body characters and hairstyle/outfit previews. Rotation changes the actual rendered view. All 17 hair options were visually reviewed, including eye-length fringes and eyebrows mounted below the hairline. Hair surfaces are tessellated before wrapping to avoid flat facets cutting into the forehead. Save commits the name/look; Cancel discards edits. Separate component tests verify that changing accounts cannot inherit another account's character or conversation.
- Native wheel scrolling is tested over a long notebook after opening and closing dialogs. Root scroll chaining no longer gets trapped by the old body overscroll rule; shared modal locks restore scrolling after overlapping drawers close.
- Custom combobox menus are exercised with keyboard selection, disabled choices and Escape inside dialogs. The sidebar has a visible collapse/expand button, contextual file navigation, full selected-item outlines and Settings beneath the profile.
- The original logo has a transparent cutout in the workspace, entry/legal screens, fallback screen, favicon and app manifest. Browser checks verify that its corner pixels are transparent.
- Real generated PDF and PPTX fixture files upload through the UI. Both source viewers support search, navigation, zoom and both themes. Custom zoom menus, round ink swatches and three stroke-size choices remain usable. PDF pen annotations, page and zoom survive navigating away and back. The native PDF aspect ratio remains intact.
- Presentation generation and design selection remain functional, including editable tables. Lecture, Editorial and Visual now change slide composition, rather than only the accent border. Tests download actual native PPTX files for all three layouts, inspect their ZIP slide content and check the PDF print document uses the selected layout. All 15 study games remain reachable. Game tile titles and descriptions pass a 4.5:1 contrast check in both themes, and flashcard answer controls and presentation annotation buttons have readable contrast. General webpage empty states and manual entry use website/video wording and a link icon. An unavailable video transcript does not trigger invented title-based notes. Website notes render headings, bold text, actual tables and supplied source images, then save that rich content into the notebook. Unsolicited AI image URLs are discarded.
- A refreshed guest session starts without the previous guest's materials while keeping a valid age/terms decision. Account restoration regressions cover unavailable Firebase, blocked local file storage and an unauthorized Google-login hostname.

AI and speech **transport responses are explicit test fixtures**. The speech fixture is playable PCM audio, not a transcript disguised as audio. These tests validate application behavior; they do not establish live Groq, Gemini, YouTube or Piper availability. Live Google login is also not claimed from mocked account tests.

## Production configuration and remaining checks

The owner confirmed that both credentials exposed in screenshots were replaced. The latest owner-provided Production configuration response after commit `1692b5a` was:

```json
{"aiConfigured":true,"providers":["groq"],"missing":[],"settingNames":["Groq_Key","Gemini_Key"],"quotaMode":"instance","geminiOptInRequired":true,"checked":"configuration only; live provider access is not tested"}
```

The updated backend accepts the confirmed `Groq_Key` and `Gemini_Key` aliases as server-only credentials. It discovers supported models, performs bounded compatible fallback, and applies a brief cooldown after rejected credentials or rate limits. Groq model rate limits do not block unrelated text models or transcription; oversized requests do not enter a provider-wide cooldown. Regression fixtures cover a busy model falling back to another listed model and a later short greeting after an oversized request. Status output contains names and booleans, never secret values.

Default `instance` quotas no longer require a Firebase service account. They reset with function instances and are not a shared deployment-wide budget. Optional `firestore` mode retains shared transactional quotas and requires both the admin credential and rate-limit secret. Gemini remains disabled until `GEMINI_ENABLED=true` is explicitly configured consistently with the app's audience and provider authorization; a key by itself does not enable it.

The executor's enforced cloud network rejects the Production URL with proxy HTTP 403. No Vercel identity, provider secrets or Firebase deployment credentials are available here. Consequently these checks remain external:

1. Confirm Vercel builds the pushed main commit successfully and `/api/status` reports the expected recognized setting names/providers.
2. Send actual guest text, image, notes/course/quiz/card and transcription requests from Production, and verify real provider responses and any configured fallback.
3. Test the existing Piper speech Space from Production, including its actual voice inventory and audio generation after waking.
4. Review and deploy the additive owner-only `users/{uid}/preferences/assistant` Firestore rule to enable cross-device conversation sync. Browser-only history remains available with a truthful sync notice if this path is denied. Do not replace unrelated live project rules without reviewing them.
5. Confirm Google login and cross-device restore with a real account on the deployed hostname.

Original uploads and generated podcast audio remain device-local, with transcripts/metadata kept in the workspace. Complex PowerPoint effects and scanned-image PDF text search are limited by the existing native viewers. Browser speech recognition and read-aloud are shown only where supported. Podcast host interaction uses recorded questions and synthesized replies, rather than an always-on real-time voice connection. It depends on microphone permission and working transcription and speech services.
