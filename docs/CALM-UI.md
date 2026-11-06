# Classio palette, notes and character update

This pass updates the existing React application. It is based on `1dc81b4` from `feat/reliable-ai-and-cloud-workspace` and uses the development branch `feat/calm-palette-notes-avatars`. Production `main` and the podcast repository are unchanged by this pass.

## What changed

- The shared light theme uses the supplied palette: deep teal `#2b6777` for actions, mist `#c8d8e4` for supporting surfaces, white, `#f2f2f2` for the page background, and restrained mint `#52ab98` accents. Dark mode uses corresponding teal surfaces and lighter readable text. User-chosen document, slide and avatar colours remain configurable.
- Notes have one document canvas. **Generate AI notes** reads the attached file, an available saved/video transcript, readable webpage text, or pasted source. Generated writing uses the existing editor's typography and width; it is appended when writing already exists. The normal undo, formatting, images, editable tables, reading view, downloads and named versions remain available. **Organize my notes** keeps an automatic version of the original writing before replacing its organization.
- Generation options, additional formatting and versions are collapsed by default. The duplicate dashboard Assistant button and the separate notes AI pane are removed. The global **AI Assistant** navigation entry and contextual **Ask this file** tool remain available.
- Original SVG characters use friendly rounded proportions, organic face shapes and soft shading. The live editor has six groups: Face, Hair, Eyes, Expression, Outfit and Accessories. It adds six face shapes, four noses, eye spacing and smile positioning while preserving existing saved style fields, custom colours and nickname. There are 17 hairstyles, 16 outfits, hats, glasses, facial hair and jewellery. The artwork is original Classio code; no Nintendo assets are included.
- Settings separates **Interface language** (English, Arabic or system default) from **AI response language** (14 named languages or automatic). Core navigation, dashboard, Settings, notes, avatar controls and Assistant controls have Arabic translations and right-to-left layout. Language changes affect subsequent requests without a refresh, including the authenticated course tool. Existing notes/files are not translated or rewritten. Some older specialized tools, help/legal content and provider error messages still use English; this is not full translation of every application screen.
- Browser preferences persist. Signed-in avatar/language settings use the existing owner-scoped Firestore profile documents. Account synchronization code is preserved; live multi-device Firebase verification requires a configured project and real sign-in.
- Native dialogs are centered, keyboard closable and sized to the viewport. The mobile navigation drawer opens from the appropriate side in Arabic. Reduced-motion settings disable character blinking.

## Source map

| Files | Purpose |
| --- | --- |
| `src/theme.js`, `src/workspace.css`, `src/entry.css`, `src/studyTools.css`, `src/documentStudio.css` | Shared palette, dark surfaces, responsive controls and uncluttered notes canvas |
| `src/NotesEditor.js`, `src/noteSource.js` | In-place AI notes and actual source resolution |
| `src/Avatar.js`, `src/AvatarEditor.js`, `src/avatar.css` | Shared original portrait renderer and live character customization |
| `src/locale.js`, `src/LanguageSettings.js` | Translations, interface direction and persistent independent language preferences |
| `src/App.js`, `src/MobileNavigation.js`, `src/Dialog.js`, `src/ChatAssistant.js`, `src/AISettings.js` | Navigation, profile/settings synchronization and translated controls |
| `src/studyTools.js`, `src/LearnMode.js`, `src/shared/course-schema.js` | Current response-language preference in shared AI and validated course requests |
| `src/Avatar.test.js`, `src/locale.test.js`, `src/noteSource.test.js`, `server/tool.test.js`, `tests/calm-ui-browser.cjs` | New regression checks |

## Validation

Executed in this environment on 6 November 2026:

- `npm run build`: successful production build, with existing warnings in the large legacy `App.js` and `noteFormat.js`.
- `CI=true npm test -- --watchAll=false --runInBand`: **45 passed** across eight suites.
- `npm run test:server`: **30 passed**, including course-language validation and forwarding; provider/database calls in these tests are mocked.
- `npm run test:browser`: existing Chromium workflow checks passed at 320×740, 390×844, 844×390, 768×1024, 1024×768, 1366×900 and 1920×1080. Includes real local eligibility cookies, refresh, navigation, theme persistence, actual missing-configuration errors, retry and chat recall, upload/view/note retention, YouTube saving/reopening and fixture-backed course editing/progress.
- `npm run test:ui`: passed palette/centered-dialog checks; same-editor notes, preserved writing, uploaded/pasted/caption source requests, navigation recall, real configuration error, avatar live editing/refresh recall, language changes/persistence, and English/Arabic layouts at 320×740, 390×844, 844×390, 768×1024, 1024×768 and 1366×900. Includes avatar tab access, Escape and reduced motion. Successful AI responses and video captions are explicit fixtures, not live external integration verification.
- `git diff --check`: passed. Source token-pattern scan found no Groq/OpenAI-shaped tokens or private-key headers; no real `.env` files are tracked. This is a targeted check, not proof against every possible secret format.

Screenshots from the new browser suite are included separately in the UI review archive. The browser is desktop Chromium using touch/viewport emulation; physical iPad/iPhone Safari was not tested.

## Service and deployment limits

The UI uses the existing trusted `/api/study` and `/api/agent` endpoints. Provider routing, fallback consent, quotas and authentication are unchanged. It does not call Groq or Gemini directly from the browser, add a provider, enable billing or put secrets in public variables.

Real provider generation, Google OAuth and multi-device cloud storage were not verified without complete production configuration. The actual local unconfigured-AI response is tested and leaves the learner's writing intact. Refer to `docs/UPGRADE.md` and `.env.example` for the existing production environment requirements.

YouTube notes require an available transcript. Websites must permit browser access (CORS); otherwise paste the source in Generation options. Private/unavailable captions, blocked websites and images without extracted text produce actionable errors. The application does not infer a video's content from its title or an image's content from its filename.

For a preview, deploy the development branch with the existing Vercel configuration, run `npm ci` and `npm run build`, and retain the existing server-only credentials. There are no dependency changes, database migrations, Firestore/Storage rule changes or podcast server changes in this pass. Review the branch before merging/deploying production.
