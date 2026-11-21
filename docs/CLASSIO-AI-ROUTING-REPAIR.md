# AI routing and production diagnostics

21 November 2026. This change fixes code defects; it does not establish that the production credentials, authentication transport, Firestore permissions, or hosted inference are working.

## Changes

The previous model selector mostly used the requested output length. Short study tasks could receive the same small model as classification, and a text model override could accidentally prevent image analysis on OpenRouter. Routing now separates workloads while preserving existing credential names and legacy text overrides:

| Work | Default routing |
| --- | --- |
| Classification, appearance JSON, short conversation | Listed fast text models, starting with Groq Llama 3.1 8B |
| Notes, questions, flashcards, presentations, longer text | Listed study models, starting with Groq Llama 3.3 70B; other eligible providers remain fallbacks |
| Image understanding | Only catalog-confirmed vision models; text overrides cannot pin this workload |
| Recording transcription | Listed Groq Whisper models; no text-only fallback |
| Creating an illustration | Separate Cloudflare FLUX.1 Schnell text-to-image route, explicitly gated |
| Background removal | Existing remove.bg service and separate confirmation |
| Podcast speech | Existing Piper TTS service; unchanged |

Server-only `AI_<PROVIDER>_FAST_MODEL`, `AI_<PROVIDER>_STUDY_MODEL`, `AI_<PROVIDER>_VISION_MODEL`, and `AI_GROQ_AUDIO_MODEL` overrides are optional. See `.env.example` for supported names. Every choice still checks the actual provider catalog, context and output limits, capabilities, provider eligibility, and free-only restrictions. The Gemini Developer API audience block remains in force.

Provider cooldowns now separate text, vision and audio. A transcription rate limit no longer disables text requests. A limit still does not rotate `packup` credentials to obtain extra quota. Cloudflare model-access, agreement and paid-plan errors likewise do not trigger credential rotation.

Cloudflare catalog properties are arrays in the documented API/model metadata. The selector now reads that shape, follows bounded pagination, skips advertised retired models and includes current catalog-backed candidates instead of depending on a single old default. OpenRouter models with image-only output cannot be selected for chat.

The previous image integration sent prompts directly from the browser to Pollinations, with placeholder alternatives and automatic fallback. Illustration requests now use `/api/images`, verified account/guest authorization, existing shared image quotas, fixed model selection, bounded prompts and returned image-byte checks. There is no external browser fallback. Image generation and sketch editing have separate capabilities; enabling text-to-image does not enable avatar sketch refinement. The slide editor explains unavailable generation and keeps uploads/search available. Existing images and drafts remain intact.

Image generation defaults off. It requires the existing Cloudflare token/account configuration, verified student-use/free-account confirmations, `AI_ENABLE_IMAGE_GENERATION=true`, and `AI_IMAGE_MODEL_APPROVED=true`. The latter requires review of the exact model/provider terms and license for this audience, and the account billing controls. No approval flags or paid settings were activated by this change. Account restrictions, refusals, exhausted allocation and invalid output preserve the original image.

Authentication failures now retain allowlisted internal phase/reason codes for SDK loading, credential configuration, app initialization, certificate retrieval, token verification and combined revocation checking. Public responses remain fixed. Logs exclude JWTs, SDK messages, raw errors, prompts and credentials. This instrumentation does not disable signature, issuer, audience, Google-provider, revocation or emulator checks. Quota configuration also rejects wrong-project and malformed administrator credentials before reporting readiness.

## Evidence and remaining production work

Local regression checks exercise the real Firebase Admin signature verifier with ephemeral RSA fixtures, provider request/response fixtures, task routing, cross-modality cooldowns, Cloudflare pagination/limits, image rejection and quota behavior, and sanitized diagnostics. Browser suites separately check real UI, editing, authentication recovery, and interrupted generation. Provider responses in those suites are fixtures, not live inference.

The handoff's production `AUTH_UNAVAILABLE` and `aiConfigured:false` observations remain separate unresolved production facts. The selected environment currently has no usable Vercel administrative binding or app credentials, and its enforced egress policy blocks the production origin. A `VERCEL_TOKEN` requirement and necessary network domains were saved to the environment draft; saving a draft does not apply it or publish it.

After authenticated Vercel access is supplied securely, inspect Production variable metadata and current deployment logs, reuse the exact primary/`packup` names, and configure only the missing requirements. Keep the existing Firebase project `classio-4378f`, shared Firestore quotas, owner-only client rules and free-only operation. A generated session secret must be written directly to secure Vercel settings. Never paste or commit secrets. Check current provider/model terms and actual account billing restrictions before enabling a provider. Do not assume a key's presence proves eligibility or free operation.

Redeploy the tested source/configuration, then inspect the fixed authentication reason/phase if verification still fails. A valid signed-in request and secure guest request must actually generate content, and a real image request must return a decodable image, before claiming those production flows are fixed. Firestore client sync and real Piper WAV synthesis require their own live checks. Do not replace missing access with weaker authentication or per-instance production quotas.

## Sources

Cloudflare's official documentation repository was inspected at commit `a1ac58e0926d99439c8b100fef550a1d5a58bf73`, including the pricing page, model metadata and error table:

- https://developers.cloudflare.com/workers-ai/platform/pricing/
- https://developers.cloudflare.com/workers-ai/platform/errors/
- https://developers.cloudflare.com/workers-ai/models/flux-1-schnell/
- https://github.com/cloudflare/cloudflare-docs/tree/a1ac58e0926d99439c8b100fef550a1d5a58bf73/src/content/workers-ai-models

Documentation does not verify this deployment's account access, licenses accepted, billing tier or provider service health.
