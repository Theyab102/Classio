# Background podcasts and study controls

22 November 2026. The owner confirmed that production generation generally works after the authentication and configuration fixes, but reported intermittent voice-question failures, podcast creation stopping during navigation, edge-aligned voice settings/errors, and counts that look read-only.

## Result

Podcast script and audio work now belongs to an account/file job rather than the mounted panel. It continues while navigating to other study sections, other files, the library, or another browser tab. A shared progress notice provides cancellation. Each file has one active job, and at most two podcasts run concurrently. Cancel, deleting material, workspace reset and account changes prevent late results from publishing. Playback and microphone capture retain their existing view lifecycle.

Background writes merge podcast fields into the current file through functional workspace updates. They preserve notes edited while work was pending and do not recreate deleted material. A new transcript is saved before synthesizing audio. Failed synthesis retains that transcript and previous audio; rebuilding saves new audio before replacing/removing the old blob. Returning reloads saved audio without generating it again.

Keep Classio open. Closing or reloading the app interrupts this browser job; saved transcripts remain available to prepare audio again. This is not a server queue and does not promise completion after the browser closes or the operating system suspends it.

Voice questions retain captured audio when transcription fails, and recognized text when the host request fails. Explicit retries reuse that input without reopening the microphone. Cancel clears it; leaving releases capture. A successful text answer remains readable even if its speech service fails.

## Routing and busy responses

Provider cooldowns separate text, vision and transcription, so a transcription outage does not disable text generation too. Routing separates short conversation/classification from study generation, including podcast scripts. Catalog-checked fast/study Groq models handle those workloads, Whisper handles transcription, and image understanding requires a vision model. Existing server text overrides stay supported and cannot pin audio or vision workloads. Optional workload overrides are in `.env.example`.

The earlier routing repair's strict zero-price OpenRouter checks, bounded Cloudflare catalog pagination and metadata parsing are included. No provider is enabled, no key is renamed, and no paid/audience confirmation is inferred. Backup keys do not supply extra quota. The separate illustration-generation changes remain on their original branch and are not part of this deployment.

The UI displays server retry intervals for rate limits. Checkpoints retain the deadline and disable resume until it expires. A failed first section says no new sections completed, rather than implying that no request was made. Free-tier capacity still depends on live service availability.

## UI and imported skills

Classio's palette, typography and workspace hierarchy are retained. Voice settings use a padded panel with responsive choices/actions. Errors remain inside their container. Question/card counts use a bordered numeric field, pencil hint, 44px minus/plus controls, and keyboard/typing support. The 1–50 range and validation are preserved. The control works on the page and in portalled dialogs. The background notice sits above the floating assistant button.

Two official GitHub skills were imported as development instructions under `.agents/skills`: Anthropic `frontend-design` and Vercel `vercel-react-best-practices`. Source revisions, observed star counts, licenses and individual SHA-256 hashes are recorded there. Only text was copied; no installer, hook, executable, SDK or app dependency was added. Main instructions and relevant React lifecycle/state guidance were read and used. Static content/credential checks supplement upstream identity and popularity; neither stars nor a scan is a malware guarantee.

## Verification

Unit regressions cover background work across unmount/remount, concurrent edits, duplicate jobs, cancellation, account switching, concurrency limits, retrying audio/transcribed questions, count controls and retry deadlines. Backend fixtures cover workload choices, capability restrictions and independent cooldowns. Firebase cryptographic regressions retain the restrictive Node loader check from A102.

`npm run test:podcast-background` exercises the built app in desktop, tablet and phone Chromium: delayed script and PCM audio across section/library/browser-tab navigation; edits preserved; old episodes kept; saved audio reloaded; typing/steppers, including portalled dialogs; voice-panel padding; contained errors and rate-limit deadlines. Provider/TTS replies are explicit fixtures. The existing podcast suite separately exercises real browser capture/Web Audio, silence detection, playback, cancellation, reduced motion and both themes. The background suite is included in CI.

Passing checks and a Git push do not establish that a Vercel deployment is Ready or that live Groq/Piper calls succeed. After deployment, repeat the owner's podcast and voice-question flows in production and inspect any remaining sanitized provider failure independently.
