# Classio

Classio is a React study workspace for organizing documents, taking notes, reviewing flashcards and building presentations. Classio is free for all users and the study workspace opens in guest mode after the 13+ check.

## Run locally

Use Node.js 22 or 24. Run `npm ci`, then `npm start` for the browser app. The guest workspace and policy pages work with the browser app alone. Use Vercel’s local development server to test the server-side AI endpoint with configured Firebase and Groq credentials and a rate-limit secret. Copy the names you need from [.env.example](.env.example).

- `npm run build` builds the browser app into `build/`.
- `CI=true npm test -- --watchAll=false` runs the focused regression tests.
- `node --test server/*.test.js` checks guest AI, transcription and provider failover with mocked providers.
- `npm run check:ai` checks the local server configuration and prints missing setting names, never their values.
- The policy URLs are `/privacy`, `/terms` and `/copyright`.

## AI providers and Vercel settings

The browser always calls the same-origin `/api/study` endpoint. Groq remains the default provider. An optional Gemini adapter supports text, inline images, conversation history and short audio recordings; keys never enter the browser bundle. With both enabled, `AI_PROVIDER_ORDER=groq,gemini` uses Gemini after a Groq rate limit, network failure or service outage. One quota reservation covers the whole request. Content-policy blocks are not retried through another provider, and truncated answers are reported as incomplete.

Set server-only `GROQ_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_JSON` and `AI_RATE_LIMIT_SECRET` in Vercel's intended environment scopes. Optional Gemini settings are `GEMINI_API_KEY`, `GEMINI_MODEL` (an available model ID from your Google AI Studio project), and `GEMINI_ENABLED=true`. Key presence alone does not enable Gemini. This integration does not provision keys, Google access or free-tier quotas. Provider free-tier limits and billing settings are controlled by their projects.

**Audience:** Classio stays free and 13+. Keep Gemini disabled unless your selected Google product's terms or authorization permit this app and audience. The existing launch guidance flags the Gemini Developer API's restriction concerning apps likely to be used by under-18s. Review the current terms before enabling it; a backend flag is not age verification. Groq remains available independently. Unconfigured AI returns a specific error inside the study tools and never prevents entering the guest workspace.

The API reserves quota before calling either provider, allows up to 20 seconds per provider, and bounds quota lookup at eight seconds. `vercel.json` assigns this Node function a 60-second budget; verify availability in your Vercel plan. Diagnostic logs include provider/error codes rather than prompts or credentials.

If a deployed study tool says AI is unavailable, open `/api/status` on that deployment. It reports missing or malformed setting names and enabled providers without returning credentials. This checks configuration only, not provider authorization, quota availability or Firestore permissions. Add the missing settings securely in the Classio Vercel project's **Production** environment, redeploy, then test an actual guest request from the site. A GitHub push deploys code; it does not create provider credentials or change Vercel's environment settings.

## Appearance and devices

The welcome page follows the device color scheme by default. Its Appearance selector offers Light, Dark and Use device setting; an explicit selection is remembered across visits. Eligibility and terms acknowledgment are also remembered in the same browser without retaining the date of birth. Guests still start a fresh workspace when the session ends.

The Library groups creation and import actions under **New**. Phones use a hamburger menu containing every study tool. Tablet and PC sidebars scroll, including account and appearance controls. A single **Ask AI** action opens the same conversation from the global header and uses the current file as context. A failed workspace download offers a reload action instead of a blank page. Reloading ends guest work, so the recovery screen explains that consequence.

## Before public launch

1. **Domain and sign-in.** Connect the chosen domain to the existing Vercel project, verify DNS and HTTPS, and add the hostname to Firebase Authentication’s authorized domains. Test Google sign-in and direct navigation to all three policy pages on that domain. The favicon uses the existing Classio logo. [Vercel domain setup](https://vercel.com/docs/domains/set-up-custom-domain)
2. **Operator and privacy notice.** Configure the legal operator name and a monitored privacy address. Confirm the live data flows, retention and deletion process, provider settings and any required privacy disclosures for the markets you serve. The policy copy is a draft until those details are verified. Check the deployed site, Vercel integrations and Google tag settings for any analytics, advertising or replay code added outside this repository. [UAE data protection laws](https://u.ae/en/about-the-uae/digital-uae/data/data-protection-laws)
3. **Age and cloud permissions.** The neutral date-of-birth check runs before Firebase loads. The date stays in memory and only the decision and terms step are kept in this browser. Ineligible decisions expire after one day so eligibility can be checked again. This is a self-declared age screen. Review whether the service is directed at children and how reports of under-13 accounts will be handled. Verify Firebase access against your actual project. [firestore.rules](firestore.rules) is an owner-only baseline for personal folders and notes. It denies client access to other collections, including legacy group data and AI rate counters. Review and test it in staging before applying it to an existing project. Also check any live Storage and Realtime Database rules. [FTC COPPA guidance](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions)
4. **AI credentials and limits.** Have an eligible service operator configure `GROQ_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_JSON` and a random `AI_RATE_LIMIT_SECRET` of at least 32 characters in Vercel. Keep provider credentials server-side and rotate any keys previously built into the browser. Test guest text, images and audio from the production origin. AI limits are shared across users and apply equally before and after sign-in. The server stores short-lived HMACs of IP addresses for quotas. Enable Firestore TTL on `aiRateLimits.expiresAt` and verify provider retention and settings. Review Groq’s conditions for serving minors. The Gemini Developer API terms exclude applications likely to be used by under-18s. [Groq Services Agreement](https://console.groq.com/docs/legal/services-agreement) · [Gemini API terms](https://ai.google.dev/gemini-api/terms)
5. **Copyright.** Set a monitored reporting address and decide how reports, counter-notices and repeat infringement will be handled. If you seek U.S. DMCA safe-harbor protection for hosted user content, complete the Copyright Office registration, publish the designated agent’s name, mailing address, phone and email on the copyright page, and maintain the designation and process. [U.S. Copyright Office Section 512 guidance](https://www.copyright.gov/512/)
6. **Marketing email.** If you send marketing email, include a working unsubscribe link, required sender information and postal address, and maintain a suppression list. [FTC CAN-SPAM guide](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business)

## Workspace behavior

Signed-in users keep an account-specific cached folder list. Cloud records are restored through the same Firebase account. The previous unscoped `classio_v2` entry stays in browser storage because it has no account identifier. If that entry contains the only copy of old offline work, recover it in the original browser before clearing site data. Original personal files are still stored separately in that browser. Guests can use all the study tools and download results. Their workspace clears on refresh or when the tab closes. The Library’s optional sign-in button imports the guest workspace into the signed-in account. Notes, progress and supported file metadata sync to Firebase. Original uploads remain on the device where they were added.

The built app serves DM Sans, Fraunces and KaTeX fonts from its own assets. General website summaries use the source website directly. If the website blocks browser access, paste the text into the study tool.

## Study tools

- **Notes** combines writing, AI note generation and lessons in one workspace. Its compact toolbar supports fonts, sizes, formatting, editable tables and images from a PDF. AI drafts are reviewed before replacing or appending to notes; replacement can be undone. The image picker searches Wikimedia Commons and shows image credits. Optional tools use a side panel on PC and a dismissible drawer on smaller screens. Named versions and HTML downloads remain available from Note options.
- **Document viewers** render original PDF pages locally and keep the same scale during page navigation. The PPTX viewer supports native text, images and tables. Both viewers support page-specific pen and highlighter strokes.
- **Presentations** preview the same generated deck in two or three layouts before applying one. Edit text, tables and images, annotate slides and export as PDF or PPTX.
- **Lessons**, inside Notes, turns a topic, document or note into lessons with a knowledge check after each lesson. Lesson explanations can be added to the editable notes without losing existing writing.
- **Review** groups flashcards and practice questions. Cards support spaced review and a TSV download. Missed cards are due again in ten minutes. Successful reviews use gradually longer intervals.
- **Record audio** can turn uploaded recordings up to 2.8 MB into a transcript and notes. It can also use compatible browsers for live dictation. Keep the original file if an upload exceeds the limit or transcription is unavailable.
- **AI Podcast** offers a quick review or deep dive, repeat playback and a script download. The optional WAV download sends the generated script to the existing `Theyab102/classio-tts` Hugging Face Space. That Space runs Piper and has an English download voice. The browser handles playback in the app. Set `REACT_APP_CLASSIO_TTS_URL` to an empty value at build time to disable the external speech integration. Test the Space after it wakes from sleep. [Classio TTS Space](https://huggingface.co/spaces/Theyab102/classio-tts)
