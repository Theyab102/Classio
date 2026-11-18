# Classio AI upgrade — implementation report

**Follow-up:** [Session recovery and dependency fixes](CLASSIO-SESSION-AUDIT-FIX.md) supersede this report's original audit and test totals. The current dependency audit reports zero findings; the historical 79 findings below describe the initial upgrade verification.

18 November 2026. Existing repository: `Theyab102/Classio`, branch `work`. The actual app is React 19/CRA with CommonJS Vercel API routes; no Next.js migration was made. Existing accounts, files, chats, editor, math, exports, themes, games and podcast service are retained. `classio-tts-server` is unchanged.

## Outcome and verification limits

Implemented authenticated multi-provider routing, exact primary/`packup` credentials, safe fallback/free-only gates, durable generation recovery, short-answer quizzes, optional browser AI/semantic search, responsive settings, credential protection and CI.

At the initial upgrade verification, build, application/authorization tests and credential scans passed, while the dependency audit reported 79 advisories (66 high, 10 moderate, 3 low), identical to an audit of the original `3fea159` lockfile against the same registry. These primarily affected inherited CRA/Jest/build dependencies. The subsequent [session and audit follow-up](CLASSIO-SESSION-AUDIT-FIX.md) resolves these findings through dependency updates and a documented local picomatch security patch, retaining CRA.

No live hosted provider, production Firebase access, Vercel variables, deployment, successful model download or GPU inference was verified. Official provider terms/pricing requests were blocked by this environment's network policy. Unverified eligibility/free operation therefore blocks activation; this report does not claim inaccessible current terms or prices were verified.

## Provider and credential status

There is no Vercel administrative connection in this task. Its actual environment-variable list cannot be inspected. The owner's declarations are recorded separately from this cloud runtime's observed absence. No values were requested, changed, deleted or copied to Git.

| Provider | Primary | Optional backup | Actual verification |
|---|---|---|---|
| Groq | `Groq_Key` | `Groq_Key_packup` | Primary declared configured in Vercel by owner; unverified. Both absent here. Adapter/credential failover tested with fixtures. |
| Gemini | `Gemini_Key` | `Gemini_Key_packup` | Both declared configured in Vercel by owner; unverified. Both absent here. Adapter retained; student inference hard-disabled. |
| Cloudflare | `CLOUDFLARE_API_TOKEN` | `CLOUDFLARE_API_TOKEN_packup` | Absent here, Vercel unknown. Adapter and account-paired failover tested with fixtures. |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_API_KEY_packup` | Absent here, Vercel unknown. Adapter, vision and zero-price restrictions tested with fixtures. |
| Ollama | `OLLAMA_API_TOKEN` | `OLLAMA_API_TOKEN_packup` | Absent here, no endpoint enabled. Protected remote adapter and SSRF rejection tested. |

No provider can be described as live operational from this evidence. Optional backups never block startup or primary-only operation.

Exact names take precedence over compatible aliases: `GROQ_API_KEY`, `GROQ_API_KEY_packup`, `GROQ_API_KEY_2`, `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `Gemini_Key_2`, `Gemini_Key2`, `GEMINI_API_KEY2`. Do not duplicate existing Vercel credentials under aliases. Cloudflare's backup uses the primary `CLOUDFLARE_ACCOUNT_ID` unless `CLOUDFLARE_ACCOUNT_ID_packup` is supplied; the token and account stay paired. Account IDs are not credentials.

Optional remove.bg, Pexels, Unsplash and Google image-search keys also use the centralized registry and support their existing name plus `_packup`. Remove.bg and Google search require verified free operation or explicit paid opt-in. File figures, local PDF processing and Commons search remain independent. The obsolete Gemini load-balancing module is not used by inference/image routes.

## Routing and server protection

Auto normally prefers compatible Groq, Cloudflare, OpenRouter and optional private Ollama. A permitted explicit mode is a preference with compatible fallback; configured provider order remains supported. Actual model catalogs, modality and estimated input/output context determine eligibility. Image bytes use a separate visual-token budget rather than being miscounted as base64 text. Text-only models never receive audio/images.

Groq candidates include its listed small Llama, larger Llama and GPT-OSS models; Whisper handles transcription and known listed vision models handle images. Cloudflare defaults to listed `@cf/meta/llama-3.1-8b-instruct-fp8` and restricts free-only choices. OpenRouter defaults to `openrouter/free` or an explicit `:free` model: all supplied catalog prices must be zero, with zero prompt/completion price constraints and underlying provider fallback disabled. Free-looking paid IDs cannot run. Live access, pricing and current limits still need verification.

Ollama requires explicit enablement, model/license/student-use/free-operation approvals and an authenticated HTTPS remote endpoint. DNS is pinned to public addresses; redirects, private destinations and credential URLs are rejected. Unauthenticated HTTP localhost works only in local development, never Vercel/production. Models are not installed in serverless functions. Qwen-family models are examples to evaluate; an environment flag does not grant a model license.

`AI_ALLOW_PAID=false` is the default. Providers require adult-operator confirmation of current student eligibility. Groq/Cloudflare additionally require verification of eligible free account operation and effective hard billing restrictions. Account billing cannot be inferred from a key. Request quotas are not a monetary cap. Gemini Developer API text/audio/image inference is blocked for this school/minor audience even with legacy enable flags; its payload/transport adapter remains implemented.

Primary credentials run first. Only key-specific 401/403 failures try a backup. Shared 429, outages and timeouts move providers instead of cycling credentials. `Retry-After`, bounded retries, credential cooldowns and exponential warm-instance circuits apply. Safety blocks stop fallback. Discovery/generation/body reading share bounded stages (normally 12 seconds with fallback, up to 18 per stage, 40 overall). Firebase authorization and study-quota lookup are bounded separately to eight seconds each; Vercel study duration is 60 seconds. Deduplication and short style/classification caches are scoped to the authorized identity. Catalogs/circuits are bounded warm-instance caches, not global persisted provider quota records.

Keyed study/image actions require a verified Google Firebase token or signed guest session. Guest access uses an HttpOnly/Secure/SameSite=Strict cookie plus CSRF token; origin checks supplement authentication. New guest identities are rate limited; refreshing a valid session retains the same identity without consuming new-session issuance quota. Missing configuration fails closed.

Vercel production requires Firestore transactional limits; instance mode is rejected. Defaults: 40 study requests per 10 minutes for both hashed IP and identity, 250 study requests per app/day, two concurrent requests per identity and eight per app. Audio/image limits are lower. Concurrency leases expire and release on completion. Quota records contain HMAC identifiers/counts/leases, not prompts or API keys. Configure TTL on `aiRateLimits.expiresAt`; existing client rules deny access to server records.

Clients cannot choose arbitrary endpoints, provide provider credentials, override paid mode or fetch another student's stored content. Input/text/image/audio budgets remain enforced. Logs contain provider/error codes and credential role, not keys or study contents. Public `/api/status` includes only general configuration, permitted provider IDs, availability and feature flags. It omits variable names, missing-variable lists, key slots, hashes and account IDs. `npm run check:ai` supplies server-only diagnostics without values. No secret inputs were added to student Settings. Privacy copy describes fallback, sessions, checkpoints and downloads without unverified retention promises.

## Durable generation recovery

Unique job IDs, frozen inputs/settings, outlines, saved sections, status and generic errors persist before/between requests. Accounts use IndexedDB plus the material's existing workspace sync. Cross-device recovery depends on healthy sync; local recovery does not require a cloud write. Guest checkpoint bytes use existing AES-GCM IndexedDB storage and its tab-session key. Navigation/refresh retain them; fresh guest tabs cannot decrypt the old session. Browser session restoration remains a documented platform limit.

The browser orchestrates bounded requests while open; this is durable resume, not an always-running background server worker. Storage failure stops generation before further provider work. Resume continues only unfinished sections; cancellation preserves completed ones. Account changes stop late output. Account/file/kind validation, IndexedDB leases, stable IDs and stale/already-applied-job checks prevent cross-user access and duplicate application. Ready drafts remain recoverable until inserted. Existing writing/decks/courses/attempts/review progress stay until validated replacement is ready. Material deletion attempts local job cleanup.

- Decks above eight slides save an outline and every slide separately. Exact counts and duplicate titles are checked. Existing templates, style choices, figures, image geometry, math, editing, undo and PDF/PPTX exports are reused.
- Long notes split source text into bounded sections and save each Markdown result, rejecting repeated long prose paragraphs. Existing editor insertion, tables, image placement, math and undo remain. Small organize/table actions keep their bounded flow.
- Courses save the outline and each validated lesson/check. Source partitioning keeps later sections and Unicode.
- Quiz/card batches save up to eight/ten items per step with exact counts, stable IDs and duplicate checks. Append/replace behavior and review progress remain.
- Short-answer quizzes add expected/equivalent concise answers, hints, explanations and saved attempts. Checking normalizes case/spacing and accepted alternatives while retaining math operators. It is a concise-answer check, not an essay grader.

Demonstrated failure at slide 15 of a 20-slide deck: slides 1–14 persist; a real browser refresh reopens the material; Resume requests only slides 15–20; all 20 survive another refresh. Durable-storage tests also cover notes, courses, quiz/card batches, cancellation, storage failure, duplicate/stale application and account changes.

## Optional open-source modules

Pinned `@huggingface/transformers@4.3.2` and `@mlc-ai/web-llm@0.2.89` are Apache-2.0. ONNX browser assets and an MIT notice are served locally. Libraries/workers load lazily; no model weights are bundled or downloaded automatically. PDF.js text/figure extraction remains local; readable PDFs do not require OCR or hosted AI extraction.

`AI_ENABLE_BROWSER_LOCAL=true` exposes explicit download consent, estimates, progress, disable/cache removal, WebGPU checks and memory guidance. WebLLM uses quantized Qwen2.5 0.5B (~400–600 MB download, ~950 MB graphics memory, 4K context) for short text. It streams internally; this change does not add hosted SSE streaming to the assistant. Image/audio/oversized local input fails clearly.

Optional English MiniLM (~25 MB weights) and multilingual MiniLM (~120 MB) support semantic search; runtime/tokenizer downloads are additional. Embeddings stay in an owner-scoped worker memory cache and clear on owner changes. Semantic matches combine with regular search across the current user's notes/course materials. Disabling local inference leaves regular search available.

Browser checks establish opt-in, no automatic model requests, worker loading, offline cache cleanup from a fresh worker with seeded model-cache entries, and visible graceful download failure when model transport is blocked. Successful model download, GPU inference and semantic-search quality remain device/network verification items. Verify actual distributed model cards/licenses before enabling downloads for production; private Ollama also requires explicit license approval.

## Credential protection

Gitleaks 8.28.0 is installed from its official release with a pinned verified checksum; the installer was executed successfully. Default rules remain enabled with exact primary/`packup`/session assignment checks. All eleven synthetic canaries in an external temporary folder are detected; no values enter Git or logs.

`check:security` scans tracked/new source, all available history and bundles. `check:staged` scans staged content and exact runtime-secret bytes. Missing scanner/errors fail closed. Build filtering was tested using random synthetic secrets in public-looking environment names; filtered builds and exact-byte/Gitleaks checks passed.

No private provider credential finding was found in current source or the 238 original commits. One existing alert is Firebase's public web configuration, reviewed by exact hash, file/context and Firebase initialization. Other Google values remain blocking. A narrow exception covers the JavaScript expression `this.vocab=new`, not real tokens or vendor files. Keep the public Firebase key restricted to intended APIs. No newly discovered private credential requires rotation from these scans; earlier owner-acknowledged screenshot exposure remains a separate event and should already have been rotated. No automatic rotation or shared-history rewrite occurred.

`.gitignore` protects private `.env.*`, `.vercel/`, exports and private key files while allowing templates. No tracked private environment file was found. Pre-commit/pre-push hooks and pinned-action CI enforce checks. Branch-protection settings remain an external GitHub configuration. Credential scans and dependency audits are separate checks; both pass after the follow-up fixes.

## Vercel variables and remaining configuration

Actual Vercel presence is unverified. Preserve the owner's existing `Groq_Key`, `Gemini_Key`, `Gemini_Key_packup` values. Do not create duplicate aliases. Optional backups are not production blockers.

Required for hosted production AI:

1. `CLASSIO_SESSION_SECRET`: random server secret, at least 32 characters. Rotation invalidates guest AI cookies.
2. `FIREBASE_SERVICE_ACCOUNT_JSON` for the existing project, `AI_QUOTA_MODE=firestore`; optional `FIREBASE_PROJECT_ID` (default `classio-4378f`) and separate `AI_RATE_LIMIT_SECRET` (otherwise session secret signs quota identifiers).
3. At least one permitted provider after authorized adult terms/account review:
   - Groq: `Groq_Key`, `AI_GROQ_STUDENT_USE_APPROVED=true`, `AI_GROQ_FREE_TIER_CONFIRMED=true`; optional `Groq_Key_packup`, `AI_GROQ_MODEL`, legacy `GROQ_VISION_MODEL`.
   - Cloudflare: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `AI_CLOUDFLARE_STUDENT_USE_APPROVED=true`, `AI_CLOUDFLARE_FREE_TIER_CONFIRMED=true`; optional `CLOUDFLARE_API_TOKEN_packup`, `CLOUDFLARE_ACCOUNT_ID_packup`, `AI_CLOUDFLARE_MODEL`.
   - OpenRouter: `OPENROUTER_API_KEY`, `AI_OPENROUTER_STUDENT_USE_APPROVED=true`; optional `OPENROUTER_API_KEY_packup`, `AI_OPENROUTER_MODEL` selecting verified zero-price free models.
   - Private Ollama: `AI_ENABLE_OLLAMA=true`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, authenticated remote `OLLAMA_API_TOKEN`, `AI_OLLAMA_STUDENT_USE_APPROVED=true`, `AI_OLLAMA_LICENSE_APPROVED=true`, `AI_OLLAMA_FREE_TIER_CONFIRMED=true`; optional `OLLAMA_API_TOKEN_packup`.
4. Keep `AI_ROUTING_MODE=auto`, `AI_ALLOW_PAID=false`. Optional `AI_ENABLE_BROWSER_LOCAL` defaults false. Gemini model names `AI_GEMINI_MODEL`/`AI_GEMINI_ADVANCED_MODEL` are retained/documented but cannot activate student inference.
5. Optional tuning: `AI_PRIMARY_TIMEOUT_MS`, `AI_PROVIDER_ORDER`, `AI_USER_REQUESTS_PER_10_MIN`, `AI_APP_REQUESTS_PER_DAY`. Existing public legal/contact/TTS and image-service names are preserved. Free-only auxiliary gates are `REMOVE_BG_FREE_TIER_CONFIRMED` and `GOOGLE_IMAGE_SEARCH_FREE_TIER_CONFIRMED`.

All names are in `.env.example`. Enter missing values securely in Vercel Production/Preview settings, never in chat or Git. Confirm Firestore admin access/TTL/private rules, provider hard spend limits and current eligibility; run server diagnostics; redeploy; test real guest/account text/image/audio. A code push does not create variables or establish deployment health. No provider was enabled to compensate for missing configuration.

## Checks and important files

Final verification: **219 frontend tests across 48 suites and 60 server tests pass**. Production build, all 11 scanner canaries, source/history/bundle scans and the final staged scan pass. Four responsive AI-upgrade browser flows pass, including a real refresh/resume of slides 15–20. Git commit/push results are recorded in the completion message. Browser AI/speech transports are explicit fixtures. Upgrade flows cover desktop, both tablet orientations and phone. Preserved-flow checks cover desktop study workflows and guest/voice/image/podcast sessions, tablet-portrait presentation editing/export, and phone notes/table/course workflows. Legacy application lint and lazy-bundle size warnings remain; no TypeScript check is configured.

Important created/modified files:

- Providers: `server/ai-settings.js`, `ai-policy.js`, `ai-adapters.js`, `ai-providers.js`, `ai-errors.js`, `ai-endpoints.js`, `credential-request.js`, `ai-configuration.js`, `study-request.js`, `image-search.js`, `image-service.js`.
- Security/routes: `server/ai-auth.js`, `ai-quota.js`, `api/session.js`, `api/study.js`, `api/status.js`, `api/images.js`, `vercel.json`.
- Recovery/integration: `src/aiAccess.js`, `aiPreferences.js`, `generationJobs.js`, `checkpointGenerators.js`, `GenerationRecovery.js`, `App.js`, `NotesEditor.js`, `courseGeneration.js`, `LearnMode.js`, `QuizStudio.js`, `FlashcardStudio.js`, `studyTools.js`, `guestWorkspace.js`, `documentMedia.js`, `imageService.js`.
- Local AI/settings/search: `src/localAi.js`, `localAiWorkerFactory.js`, `localAi.worker.js`, `AISettings.js`, `aiUpgrade.css`, `LibraryHome.js`, `Legal.js`, `public/local-ai-runtime-license.txt`, manifest/lockfile, `scripts/build.cjs`.
- Safeguards/docs: `.env.example`, `.gitignore`, `.gitleaks.toml`, `.githooks/`, `.github/workflows/security.yml`, `scripts/install-security-tools.sh`, `security-check.cjs`, `security-canaries.cjs`, `check-ai-config.cjs`, README and this report.
- Tests: provider/auth/shared-quota/session/endpoint tests, generation/settings/local-cache tests, updated authenticated fixtures/duplicate-free generation fixtures, and `scripts/ai-upgrade-browser.cjs` plus existing browser suites.

Reusable cloud installation/start instructions now include the pinned scanner and AI-upgrade checks. They were saved as a draft; this does not apply or publish a new environment snapshot. Review/save in environment settings and publish through the product when desired.
