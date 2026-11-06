# Classio

Classio is a React study workspace for organizing documents, taking notes, reviewing flashcards and building presentations. Classio is free for all users and the study workspace opens in guest mode after the 13+ check.

## Run locally

Use Node.js 22 or 24. Run `npm ci`, then `npm start` for the browser app. The guest workspace and policy pages work with the browser app alone. Use Vercel’s local development server to test the server-side AI endpoint with configured Firebase and Groq credentials and a rate-limit secret. Copy the names you need from [.env.example](.env.example).

- `npm run build` builds the browser app into `build/`.
- `CI=true npm test -- --watchAll=false` runs the focused regression tests.
- `node --test server/study.test.js` checks guest AI and transcription routes with a mocked provider.
- The policy URLs are `/privacy`, `/terms` and `/copyright`.

## Before public launch

1. **Domain and sign-in.** Connect the chosen domain to the existing Vercel project, verify DNS and HTTPS, and add the hostname to Firebase Authentication’s authorized domains. Test Google sign-in and direct navigation to all three policy pages on that domain. The favicon uses the existing Classio logo. [Vercel domain setup](https://vercel.com/docs/domains/set-up-custom-domain)
2. **Operator and privacy notice.** Configure the legal operator name and a monitored privacy address. Confirm the live data flows, retention and deletion process, provider settings and any required privacy disclosures for the markets you serve. The policy copy is a draft until those details are verified. Check the deployed site, Vercel integrations and Google tag settings for any analytics, advertising or replay code added outside this repository. [UAE data protection laws](https://u.ae/en/about-the-uae/digital-uae/data/data-protection-laws)
3. **Age and cloud permissions.** The neutral date-of-birth check runs before Firebase loads. The date stays in memory and only the decision and terms step are kept for the tab. This is a self-declared age screen. Review whether the service is directed at children and how reports of under-13 accounts will be handled. Verify Firebase access against your actual project. [firestore.rules](firestore.rules) is an owner-only baseline for personal folders and notes. It denies client access to other collections, including legacy group data and AI rate counters. Review and test it in staging before applying it to an existing project. Also check any live Storage and Realtime Database rules. [FTC COPPA guidance](https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions)
4. **AI credentials and limits.** Have an eligible service operator configure `GROQ_API_KEY`, `FIREBASE_SERVICE_ACCOUNT_JSON` and a random `AI_RATE_LIMIT_SECRET` of at least 32 characters in Vercel. Keep provider credentials server-side and rotate any keys previously built into the browser. Test guest text, images and audio from the production origin. AI limits are shared across users and apply equally before and after sign-in. The server stores short-lived HMACs of IP addresses for quotas. Enable Firestore TTL on `aiRateLimits.expiresAt` and verify provider retention and settings. Review Groq’s conditions for serving minors. The Gemini Developer API terms exclude applications likely to be used by under-18s. [Groq Services Agreement](https://console.groq.com/docs/legal/services-agreement) · [Gemini API terms](https://ai.google.dev/gemini-api/terms)
5. **Copyright.** Set a monitored reporting address and decide how reports, counter-notices and repeat infringement will be handled. If you seek U.S. DMCA safe-harbor protection for hosted user content, complete the Copyright Office registration, publish the designated agent’s name, mailing address, phone and email on the copyright page, and maintain the designation and process. [U.S. Copyright Office Section 512 guidance](https://www.copyright.gov/512/)
6. **Marketing email.** If you send marketing email, include a working unsubscribe link, required sender information and postal address, and maintain a suppression list. [FTC CAN-SPAM guide](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business)

## Workspace behavior

Signed-in users keep an account-specific cached folder list. Cloud records are restored through the same Firebase account. The previous unscoped `classio_v2` entry stays in browser storage because it has no account identifier. If that entry contains the only copy of old offline work, recover it in the original browser before clearing site data. Original personal files are still stored separately in that browser. Guests can use all the study tools and download results. Their workspace clears on refresh or when the tab closes. The dashboard’s optional sign-in button imports the guest workspace into the signed-in account. Notes, progress and supported file metadata sync to Firebase. Originals with completed private Firebase Storage uploads can be recalled on another device signed into the same Google account. Older browser-only originals must be recovered or uploaded from the original device first.

The built app serves DM Sans, Fraunces and KaTeX fonts from its own assets. General website summaries use the source website directly. If the website blocks browser access, paste the text into the study tool.

## Study tools

- **Notes Studio** supports formatting, editable tables, uploads and images from a PDF. AI can organize a draft or build a table for review. The image picker searches Wikimedia Commons and shows the image credits. Google Images opens as a separate manual search.
- **Document viewers** render original PDF pages locally and keep the same scale during page navigation. The PPTX viewer supports native text, images and tables. Both viewers support page-specific pen and highlighter strokes.
- **Presentations** preview the same generated deck in two or three layouts before applying one. Edit text, tables and images, annotate slides and export as PDF or PPTX.
- **Learn Mode** turns a topic, document or note into lessons with a knowledge check after each lesson.
- **Study Cards** offers spaced review and a TSV download. Missed cards are due again in ten minutes. Successful reviews use gradually longer intervals.
- **Record audio** can turn uploaded recordings up to 2.8 MB into a transcript and notes. It can also use compatible browsers for live dictation. Keep the original file if an upload exceeds the limit or transcription is unavailable.
- **AI Podcast** offers a quick review or deep dive, repeat playback and a script download. The optional WAV download sends the generated script to the existing `Theyab102/classio-tts` Hugging Face Space. That Space runs Piper and has an English download voice. The browser handles playback in the app. Set `REACT_APP_CLASSIO_TTS_URL` to an empty value at build time to disable the external speech integration. Test the Space after it wakes from sleep. [Classio TTS Space](https://huggingface.co/spaces/Theyab102/classio-tts)

## Critical upgrade branch

See [the upgrade and deployment guide](docs/UPGRADE.md) for the shared Groq/Gemini backend, signed eligibility, private cross-device file recall, migration behavior and remaining work. See [validation results](docs/VALIDATION.md) for checks actually executed. This branch is not the complete podcast/avatar redesign.
