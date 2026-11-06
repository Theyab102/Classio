# Classio critical functionality and cloud workspace upgrade

This development branch starts at Classio `799b891` and speech server `135098c`.
The previously described Blue Workspace ZIP was not attached to this task and
has not been merged. This is an implementation of the critical functionality
and persistence foundation, not the complete requested podcast/avatar redesign.
No production deployment, Firebase rule change or user-data deletion was performed.

## Confirmed findings

- The React 19 / CRA 5 app uses component state for navigation, inline styles and
  CSS, Firebase Google authentication, Firestore, and local IndexedDB originals.
- `/api/study` formerly returned a generic 503 unless Groq, Firebase Admin and a
  32-character quota secret were all configured. These settings are absent in
  this machine; production Vercel settings are not accessible here.
- Google popup sign-in exists. Age/terms decisions were session-storage strings.
- Original file bytes were never uploaded to cloud storage. Syncing the folder
  list could not make the original PDF/PPTX/text file available on another device.
- The named-folder link action discarded the pasted URL. Missing video transcripts
  triggered title-based generation. No Signum references were found in tracked
  app source/dependencies.
- The speech repository also retains an unused legacy `App.js` snapshot that expected a public `REACT_APP_GROQ_KEY`. Its provider calls now use `/api/study`. If a real key was ever used in that older frontend build, rotate it; this task could not inspect old deployment bundles or logs. No literal private key was found in the targeted source/history scan.
- The Python service blocks its async route during synthesis, exposes exception
  text, and can retain an incomplete model/configuration pair after a failed download.

## Implemented architecture

- `server/ai-providers.js`: shared official Groq and Gemini REST adapters. Text,
  supported vision requests and the existing specialist Groq transcription route
  stay server-side. A provider preference applies throughout the app.
- Auto prefers configured Groq, otherwise configured Gemini. One alternate attempt
  is permitted only with explicit fallback consent and only for temporary errors
  or rate limits. Invalid keys/models never trigger alternate-provider retries.
  The response identifies the provider and whether fallback occurred.
- Gemini requires a key, an explicitly selected model, `GEMINI_ENABLED=true` and
  `GEMINI_FREE_TIER_VERIFIED=true`. These flags require an operator's actual review
  of current official pricing, data-use/audience terms and Google project billing.
  They do not themselves enforce Google's billing controls. Existing documentation
  describes a 13+ audience; do not assume this audience is eligible for Gemini.
- `/api/agent` exposes one validated, account-only action: `create_course` for a
  file owned by the verified Firebase token UID. It validates generated lessons,
  handles request IDs and acknowledges saving only after a database transaction.
  It cannot choose an arbitrary database path, execute commands, or delete data.
  The Assistant's Create course action opens this real workflow; model-driven
  autonomous tool selection and other agent tools are not implemented.
- `/api/access` signs minimal eligibility decisions into a year-long HttpOnly,
  SameSite cookie, secure on production HTTPS. Dates of birth are not persisted.
  Verified accounts can restore the decision on another device after Google login.
  The server-only verification record is outside client-writeable collections.
  This remains self-declaration, not identity or documentary age verification.
- Firebase web initialization is shared. Google popup/redirect handling, persistent
  auth and specific errors remain with the existing provider. A pending guest
  import is retained through an OAuth redirect. Production OAuth is unverified.
- Cloud workspaces use owner-private `users/{uid}/folders` and `files` collections,
  with realtime listeners and field patches. This avoids replacing an entire
  workspace from another device. Simultaneous edits of the same field still use
  last-write behavior; this is not collaborative document editing.
- Original uploads use Firebase Storage at `users/{uid}/files/{fileId}/{sha256}`.
  Versions are immutable. Reads use the authenticated Storage SDK, not public
  download-token URLs. Downloads verify SHA-256. The per-file limit is 20 MiB.
  No storage bucket or billing plan was provisioned or enabled automatically.
- Avatar configuration and theme preferences have owner-scoped profile records.
  Assistant conversations are owner-scoped Firestore documents, with a tab-local
  draft cache. New chats, Markdown/math, image questions and retry are supported.
- Blue/slate theme tokens, responsive navigation drawer and native dialog overlays
  cover the upgraded interfaces. The existing SVG avatar and legacy study tools
  remain in place. This does not establish a complete accessibility/dark-mode audit.

## Data migration and recovery

Migration copies legacy `users/{uid}.folders` into per-folder/per-file documents
only if each destination does not already exist. A completion marker prevents
repeat migration. The legacy arrays are retained. Deletions in the new schema
use tombstones; no automatic storage cleanup runs.

Old originals have no cloud copy. Open the app in the original browser/account
and explicitly recover that account's matching original using the file viewer's
recovery action, or upload the original again. Recovery preserves the old browser
entry and makes a new account-scoped cache entry. After a completed cloud upload,
the same Firebase Google account UID can recall it on another device. Matching
email text alone never grants access to another UID's files.

A pending upload/save is not a cloud backup. Keep the tab open until syncing
finishes. Failed metadata writes remain in the account-scoped local pending cache and have
a retry action. An original upload failure does not block notes/course metadata
sync; its separate warning and retry state never claim the original is backed up. Review migration against staging data before production use.

## Configuration and deployment

1. Review the two development branches/source folders independently. Install with
   `npm ci` using Node 22/24. `npm start` now runs CRA and a local API on port 3001.
   The local API generates a development-only cookie key if none is supplied;
   set a stable `CLASSIO_SESSION_SECRET` locally if verification should survive
   restarting that API process. Runtime processes are not part of an environment snapshot.
2. In Vercel's server environment configure `GROQ_API_KEY`, the existing Firebase
   service account as `FIREBASE_SERVICE_ACCOUNT_JSON`, and `AI_RATE_LIMIT_SECRET`
   (at least 32 random characters). Prefer a separate `CLASSIO_SESSION_SECRET`.
   Reuse existing valid values; never paste keys into chat, browser code or Git.
   Review the model catalog in the actual Groq project; set vision separately.
3. For Gemini, create a key securely at https://aistudio.google.com/apikey if needed, selecting an appropriate Google Cloud project. Do not enable billing or paid services without separate approval. Verify current
   supported models, pricing/free quotas and audience/data-use terms at
   https://ai.google.dev/gemini-api/docs/models,
   https://ai.google.dev/gemini-api/docs/pricing and
   https://ai.google.dev/gemini-api/terms. Then configure `GEMINI_API_KEY` and
   `GEMINI_MODEL` server-side and opt in with the two flags. No paid Gemini service
   or Google billing change was activated in this task. Real generation with
   either provider must be tested after configuration.
4. Preserve the existing Firebase project and enable Google in Authentication.
   Authorize the final Vercel/custom domain. If overriding the public auth domain,
   follow Firebase's official redirect-helper/custom-domain setup, especially for
   Safari. Verify popup, redirect, refresh, logout and account linking on production.
5. Configure a private Firebase Storage bucket in the same project and its CORS
   policy for the actual app origins. Set its exact name in
   `REACT_APP_FIREBASE_STORAGE_BUCKET` at build time. Check current Firebase Storage
   plan/billing requirements and quotas before enabling it; this task did not
   provision a bucket or authorize a paid plan. Deploy reviewed `firestore.rules`
   and `storage.rules` to staging first. A CLI example for an authorized operator:
   `firebase deploy --only firestore:rules,storage --project YOUR_PROJECT_ID`.
   Do not switch projects or erase existing records. Check any existing Storage,
   Realtime Database and group rules independently before combining rule changes.
6. Enable Firestore TTL on `aiRateLimits.expiresAt` if not already configured.
   Default AI caps remain 18 text requests / 10 minutes per connection/account and
   250 text requests / day globally, with stricter transcription caps. Guest
   study requests retain signed eligibility + quotas; private tool actions require
   verified Google authentication. These controls are not a guarantee against all abuse.
7. Rebuild/redeploy Vercel after changing server settings or public build settings.
   Keep `REACT_APP_CLASSIO_TTS_URL` pointed at the actual speech deployment.
8. For the Python service, use Python 3.11.9 and `pip install -r requirements.txt`.
   Start `uvicorn server:app --host 0.0.0.0 --port 8000`. Configure
   `CLASSIO_ALLOWED_ORIGINS` for the production and authorized preview origins.
   The pinned Piper voice release must be reachable from Hugging Face and model
   download hosts. Cached artifacts are checked against Piper's bundled manifest;
   do not bypass verification if a voice fails. Generate real WAVs before deploying.

## Validation status

See `VALIDATION.md` for executed tests and blocked/unrun checks. In particular,
SDK/rules tests could not execute because the emulator download host is blocked.
The source is prepared for cross-device recall, but this has not been proven
against a real or emulated Firebase backend in this task. No production readiness
claim is made while those checks and configuration remain unresolved.

## Remaining requested work

- Real Groq/Gemini responses and free-tier/model eligibility verification.
- Production Google login/linking and two real devices retrieving the same upload.
- Executed Firestore/Storage rules, migration and concurrent-save integration tests.
- A complete Mii-inspired avatar visual redesign, beyond persistence of the existing editor.
- Two-host podcast script/audio assembly, background jobs, measured caption timing,
  animated speaking hosts and the new full player/customization workflow.
- A full review of every legacy modal, dropdown, dark-mode screen and touch layout;
  actual iPhone/iPad Safari and comprehensive WCAG 2.2 AA testing.
- Autonomous AI tool selection and additional authorized tools beyond course creation.

These are remaining work, not implemented or tested features of this delivery.
