# Reliable study generation

Generation now belongs to the current account and material, rather than the mounted tool panel. Notes, quizzes, flashcards, courses and presentations can finish while the learner uses another Classio menu. Existing podcast background work remains supported. A global notice shows the file, saved-section percentage and cancellation control.

## Capacity and recovery

A study request runs through one per-account browser queue. On HTTP 429 it honors `Retry-After` (seconds or HTTP date, including daily resets), waits without consuming the request timeout, then repeats only the unfinished request. Checkpoints record waiting state and renew the cross-tab lease during long waits. The UI shows a calm queue message, saved sections and real progress; it does not claim a quota percentage or an estimated completion time. Voice synthesis independently waits for its own capacity reset and repeats the unfinished turn.

Closing a generation panel or changing menus keeps work running. Explicit Cancel pauses it. Account changes, material deletion and workspace reset cancel work and reject late writes. Completion uses functional updates against the latest file, preserving concurrent notes, linked sources and other generated tools. New presentation proposals are kept for review if the learner edited the current deck during generation.

This is browser background work: Classio must remain open. Browser/OS suspension can delay timers. A reload restores checkpoints, and opening the relevant study tool resumes unfinished work automatically after its recorded reset. Explicitly paused work needs Resume. Closing the browser does not run a Vercel worker. Authentication, missing configuration and content errors remain actionable errors; they are not hidden in an endless quota retry.

## Source coverage and graphs

New notes use a deterministic page/question plan instead of spending a model call on an outline. Every readable PDF page is retained; dense pages split at question boundaries, at most three questions per section when labels are available. Logistics-only lines are excluded. Linked answer keys are references rather than duplicate teaching courses, and matching question labels select later answer-key content. Each section must include its source question labels; missing questions, whole-section duplication and invalid graph specifications trigger a bounded repair of that section. Reusable definitions are allowed.

Prompts read file names and actual source text, respect learner instructions, teach prerequisites and worked examples, and avoid assuming calculus from a grade label. End behavior uses graph shape unless advanced methods are explicitly taught/requested. Diagram pages can attach an original PDF page to a vision-capable request. Unavailable or unreadable originals must be disclosed. Source content remains untrusted reference material.

Arithmetic graph specifications are parsed locally without JavaScript evaluation. The graph has axes, ticks, a title and a safe sampled function. Notes store the resulting local PNG in rich content; ordinary text used as AI context excludes base64 images. Assistant/lesson Markdown renders validated graphs as SVG. These are equation plots, not invented diagrams or paid image generation. Complex unreadable geometry still requires an available vision model and a readable original.

## Provider roles and shared quotas

Server-owned routing selects fast models for classification, style and planning; study models for teaching and mathematical reasoning; vision models for image input; and transcription models for audio recognition. Eligible free providers can fall back independently. Fast work prefers Cloudflare when eligible; study work prefers Groq, then eligible alternatives. Model catalogs must confirm modality, context, output capacity and free eligibility. `AI_FAST_PROVIDER_ORDER`, `AI_STUDY_PROVIDER_ORDER`, `AI_VISION_PROVIDER_ORDER` and `AI_CONVERSATION_PROVIDER_ORDER` can override role order server-side.

Gemini Developer API remains unavailable for the configured school/minor audience under the existing provider policy. A second key for the same provider is a credential backup, not another quota pool. It is tried only after credential rejection, not to evade rate limits. No paid-provider approval, audience confirmation or provider credential is invented by this change. Routing avoids an extra Gemini prompt-writing call for every request, which would increase latency and quota use.

## Reviewed skills

Added `llm-evaluation` and `rag-implementation` reference skills from `wshobson/agents`, pinned to commit `51b9ae3ac68d9c236549a9b57f5dca7e6c0c168c` (42,821 GitHub stars observed 23 November 2026). Their source-grounding and evaluation guidance informs coverage, retrieval and failure tests. Only Markdown instructions and MIT licenses were imported; no example frameworks, scripts or evaluation services run in production. Hashes and review notes are in `.agents/skills/provenance.json`. Existing reviewed frontend, accessibility, composition and React skills remain in use.

Validation uses deterministic provider fixtures for long-PDF coverage, actual browser extraction, rate-reset timing, graph rendering, background completion, concurrency, cancellation, account boundaries and deletion. Fixture success does not establish live production provider availability or the correctness of an unseen revision pack.
