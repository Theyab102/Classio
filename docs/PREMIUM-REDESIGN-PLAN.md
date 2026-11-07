# Classio premium workspace implementation

## Architecture and history

The current project is React 19 with Create React App, Firebase Authentication/Firestore, local IndexedDB uploads, TipTap rich notes, local PDF.js/PPTX viewers, and Vercel Node API routes. `src/App.js` owns existing account restoration, folders, personal files, presentations, recording, and study games. Preserve those handlers and data shapes while moving new interfaces into focused components.

Git history at `8a7c3e2` identifies the four removed home shortcuts exactly: **Blank document**, **Record audio**, **Upload document**, and **Website link**. Their handlers still exist. Restore visible cards connected to those handlers. Existing study games and presentations remain available.

## Implementation sequence

1. **Provider configuration and secret handling.** Support the confirmed case-sensitive server aliases `Groq_Key` and `Gemini_Key`; preserve canonical names. Strip legacy public provider variables from the frontend build. Keep credentials out of diagnostics, source, and logs. Bound failover and improve provider error messages. Do not use credentials exposed in screenshots; the owner confirms replacements were saved.
2. **Workspace and Library.** Restore the four shortcuts. Add original Classio visual structure, real search/filter/sort, document previews, folders and recent activity derived only from actual opens. Keep CRUD handlers and destructive confirmations.
3. **Assistant.** Keep a collapsible right-side composer and a navigation entry. Use a desktop side panel that reserves content space and a mobile drawer. Share history across navigation, isolate account sessions, and send only the active material/context explicitly selected by the user.
4. **Avatars and profile.** Keep Face/Hair/Outfit/Details and legacy option fields. Render original rounded Three.js character meshes with lighting, shadows and full-body rotation. Replace text-only selections with live thumbnail grids. Stage edits locally, provide randomize/reset/save/cancel, and persist only committed choices through existing account storage.
5. **Notes, lessons, quizzes and cards.** Preserve TipTap editing, source figures, tables, versions and downloads. Add equations, polished learning layouts, real generated quizzes with validated schemas, hints/explanations and saved progress. Keep legacy games available. Improve flashcard editing and review without resetting saved cards.
6. **Documents and podcasts.** Preserve original document proportions and annotations, add supported document search and zoom/navigation polish. Use real synthesized audio with metadata-driven duration/seek/transcript alignment, clear errors and saved episode/playback data. Keep recording available. Support recorded spoken interruptions with transcription, a grounded spoken answer and episode resumption; do not present this as an always-on real-time voice service.
7. **Themes and interaction.** Use light blue, readable white/neutral and balanced dark surfaces, original educational illustrations, keyboard focus, reduced-motion rules and responsive layouts.
8. **Validation and release.** Run component/server regressions, build and meaningful browser tests on phone/tablet/desktop in both themes. Test preserved data and feature routes. Push the reviewed changes to `Theyab102/Classio` main, as authorized. Verify Production only where access is actually available.

## External verification limits

At task start, the selected cloud executor has no Vercel identity or server secrets. Its enforced outbound policy rejects `classio-eta.vercel.app` with proxy HTTP 403. No Vercel or shared-browser control tools are callable. Screenshots confirm variable names but do not prove live values, model access, quota storage, or a completed redeployment. Record live AI as unverified until an actual production response is obtained.

## Preservation contract

No account or library data reset; no age/onboarding reset; guests retain access to every study tool; Google sign-in remains optional; no tracking/replay added; fonts remain self-hosted. Existing uploaded originals remain device-local. New fields are additive and tolerate older records.

## Implementation outcome

The eight implementation stages above are complete in code. The four original home actions are restored; existing presentations, recording and study games remain available. The Library, assistant, visual character editor, notes, lessons, quizzes, flashcards, podcasts and source views now use the original Classio blue/neutral design. Reviewable validation results and remaining external checks are in [REDESIGN-VERIFICATION.md](REDESIGN-VERIFICATION.md).

The confirmed Production setting aliases are supported server-side. Default instance quotas remove the unnecessary Firebase-admin dependency for guest AI. Shared Firestore quotas remain available explicitly and must be configured correctly; instance counters are not a global spending cap. Gemini retains its existing explicit opt-in for the app's audience. Production credentials and live provider access cannot be verified from this environment, and Firebase rules cannot be deployed without the relevant project access.
