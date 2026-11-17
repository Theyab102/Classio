# Guest sessions, voice conversations and study feedback

Continues from `6ceffe6eea6c76708060ca06a67db7ca2968a44b` in Classio.
The speech service source does not need a change for this round.

## Guest work

Guest navigation no longer runs the blank-note deletion filter. A guest can
leave a file, return to the Library, reopen it and refresh the page without
losing writing, files, generated cards, course progress or podcast metadata.
An explicitly ended session starts empty. Signed-in cleanup also recognizes
saved voice notes as content.

Workspace metadata, chat text/drafts and the selected avatar use tab-scoped
`sessionStorage`, separate from account caches. Original uploads, voice recordings
and podcast audio use temporary IndexedDB entries encrypted with AES-GCM; their
key exists only in that tab session. Reload restores the originals. A fresh tab
does not inherit a closed guest workspace or its key. Signing in retains the
existing intentional guest-import flow; ending a guest session clears its metadata
and key and removes its temporary entries.

Browsers do not provide a reliable "the user clicked X" event distinct from
reload. Clearing storage in `beforeunload` would recreate the reported bug.
Session storage supplies the usual tab lifetime; browser **Restore closed tab**
or **Continue where you left off** can restore a browsing session, including its
session storage. The website cannot promise to override those browser features.
**End guest session** provides explicit clearing.

Abandoned encrypted cache entries can be collected after a day. Web Locks protect
other open tabs during collection, and a session lease protects reloads. Where
Web Locks are unavailable, automatic cross-tab collection is skipped; ciphertext
remains inaccessible to a new session and browser storage clearing removes it.
Storage failures remain visible; content is not silently moved to a permanent
guest account cache.

## Recording and image questions

The assistant, Library recording dialog, voice notes and Voice Answer game now
use `getUserMedia` plus `MediaRecorder`, rather than depending on Chrome/Edge's
remote SpeechRecognition service. Recording, permission waiting, transcription,
cancellation and errors are explicit states. Late permission grants after cancel
release their tracks; leaving the control aborts transcription and stops capture.
Permission denial, a missing/busy microphone and unsupported recording produce
actionable messages.

Assistant speech becomes an editable draft with an audio preview. It does not
send automatically: the learner chooses Send. Voice notes likewise expose the
transcript, optional organization into notes and an intentional save. The original
audio remains available when transcription/organization fails. Voice-note clips
allow up to ten minutes at a requested low speech bitrate, bounded by the existing
2.8 MB transcription limit; assistant/game answers use shorter clips. Long
recordings can be kept as originals or recorded in sections.

Image questions use a separate image preparation budget (up to 1800 pixels on the
long edge), preserving more detail than inline note thumbnails. PNG, JPEG, WebP,
GIF and browser-decodable AVIF are accepted; unsupported HEIC gets an export hint.
Send waits until image preparation completes, and clipboard image paste is
supported. An image question uses its image and existing notes without waiting
for an unrelated original document to parse. The resulting multimodal payload
still passes the server's existing validation and vision routing. Attachment
image bytes remain excluded from saved conversation history.

## Podcasts

**Ask the hosts aloud** is now the primary action immediately below the host
characters, above playback/download/transcript utilities. It pauses the episode
at the real playback position. The voice flow records, detects a pause,
transcribes, sends the question and synthesizes the answer. Question/answer text
and typed input stay collapsed by default. The answer now starts playing only
after its audio player mounts; the previous timer could run before that mount.

A learner can interrupt the answer or pending generation and ask again. Old
recording callbacks cannot stop the new microphone session. By default the dialog
listens for another question when an answer ends. **No More Questions** resumes
the episode at the saved interruption position. The existing checkbox also lets
the learner choose automatic episode resume after an answer. Cancel and close
release the microphone and requests; failed speech synthesis keeps the answer
available to read/retry.

## Games, avatars and branding

Opening games without cards prepares one source-grounded card set automatically
and saves it to the file's Flashcards too. Existing cards are reused. Empty or
unreadable material asks for source content instead of inventing it from a
filename. Preparation supports cancellation, failure and intentional retry.

Original, locally synthesized tones provide correct/wrong/completion feedback in
study games, quizzes, flashcard review and lesson checks. **Sound on/off** is
available in game headers and **Settings → Study feedback**, with a remembered
preference. It requires no external sound assets.

Avatar Hair, Details → Hijabs, and Details → Accessories include Classic,
Wrapped, Sport and Draped hijabs, with an independent cloth color. Hair is hidden
under the scarf; choosing a hat or hairstyle removes the scarf. A hijab pin and
scarf brooch supplement the existing accessory catalog without changing existing
option indices. Changes use the normal preview/Undo/Save/session/account paths.
The original book logo brightens and softens its saturation in dark mode across
the welcome, policy and workspace pages, retaining the original artwork.

## Validation and production limits

```sh
CI=true npm test -- --watchAll=false --maxWorkers=2
npm run test:server
npm run build
npm run test:sessions
npm run test:workflows
npm run test:notes
npm run test:browser
```

The new session suite exercises actual Chromium encryption/storage, uploaded
image decoding, encoded microphone capture, muted/unmuted Web Audio feedback,
WAV playback, podcast interruption/resume and tab isolation. It covers desktop,
landscape tablet, portrait tablet and phone; it also inspects hijab selections
and restoration. AI, transcription and synthesis responses are explicit fixtures.
Physical microphone quality, mobile Safari and live model/voice services require
separate device/deployment verification.

In this environment, `check:ai` reports no configured local provider and the
production `/api/status` request is blocked by the environment proxy (CONNECT
403). No deployed credentials or production settings were changed. Those limits
prevent a claim that live transcription, vision and TTS providers were verified.
The existing production Firebase rules issue is separate and is documented in
the [previous follow-up](CLASSIO-INTERFACE-SYNC-FOLLOWUP.md).

Logs: `/workspace/.onboarding/session-*.log`. Screenshots:
`/workspace/artifacts/classio/guest-voice/`. These artifacts and fixtures are not
part of the shipped browser assets.

Current-instance results: **41 frontend suites / 172 tests**, **42 API tests**
and **6 speech regression tests** pass. The production build passes with the
remaining legacy lint warnings. The four-device session/voice suite, four-device
workflow suite, four-device notes/table/course suite and focused phone layout/font
check pass. The voice suite includes saving/reloading actual recordings and
reviewing a spoken game answer before sending it for grading.
The broad desktop/phone suite also passes, including responsive checks in both
themes and 125%/150% zoom emulation. An additional phone run verifies that a
restored microphone recording actually starts playing and advances its position.
