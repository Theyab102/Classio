# Tablet input, feedback volume and AI fallback

The annotation toolbar now keeps Undo and Redo visible beside the drawing tools,
including in Scroll mode. The controls use curved-arrow icons, labels, disabled
states and 44px minimum targets. Existing page-scoped history and keyboard
shortcuts remain available.

Source PDFs and PowerPoints now distinguish drawing from navigation. A stylus or
mouse draws with the selected tool; fingers navigate by default. One or two
fingers drag vertically to scroll and swipe horizontally to change one page.
A gesture commits once when the original finger group starts lifting. Changing
finger count rebases its origin, cancellation discards it, and a pinch cannot
be mistaken for a page turn. Zoomed material pans horizontally before a new
swipe at its edge can turn the page. Inputs and toolbar controls are excluded.

Drawing options includes explicit **Draw with finger**. When enabled, one finger
can draw; adding a second finger abandons the provisional stroke and navigates.
Pointer cancellation discards its preview instead of saving an accidental line.
Resize and animation callbacks read current ink history, and redraws clear the
previous bitmap even when layout bounds are temporarily zero. Fit width now uses
the actual viewport padding; the old fixed subtraction left a narrow horizontal
overflow that could intercept page swipes. The PowerPoint
annotation toolbar sits above the slide viewport, so fitting or zooming a page
does not shrink the controls.

Correct, wrong and completion feedback use a peak envelope of 0.21 instead of
0.07 (three times the amplitude, about +9.5 dB). The sounds keep their short fades
and on/off setting. Browser verification measures actual Web Audio output and
checks that the boosted signal does not clip.

The server keeps Groq first by default. Successful primary calls consume neither
Gemini key. A whole provider stage now has a deadline covering catalog discovery,
generation and body reading, so a stalled connection or stalled JSON body cannot
use all the time available for fallback. The stage defaults to 12 seconds when
another provider/key is available, within the existing 40-second total deadline.
`AI_PRIMARY_TIMEOUT_MS` can tune the stage between 1000 and 18000 milliseconds.
Requests run sequentially; an expired stage is aborted and cannot start late
model requests. Network failures cool down for 30 seconds; skipped requests do
not extend that cooldown indefinitely. Safety blocks and non-retryable request errors stop
without fallback.

Free Gemini work selects a slot using estimated work reserved during the current
minute, including concurrent requests. Ties favor slot 1 for learning content and
slot 2 for conversation, classification, style, vision and transcription. Image
generation shares the same balancing pool. Provider order/model/quota-group
changes also change the request-cache identity. Duplicate in-flight requests
and the existing short cache for style/classification still conserve calls.

These estimates and cooldowns are local to a warm server instance. They are not
provider billing counters or a promise of unlimited free generation. Keys from
one Google project retain a shared quota cooldown; a second key does not create
more project allowance. Independent quota-group settings are appropriate only
for genuinely independent provider projects. No credentials or deployment
environment settings were changed or committed.

## Validation

- Frontend: 44 suites, 196 tests passed.
- API/server: 51 tests passed, including stalled connections, stalled generation
  bodies, stalled catalog bodies, primary recovery after cooldown, both free
  slots, simultaneous reservations, shared-project rate limits and image routing.
- Production build passed. Existing unrelated warnings remain.
- The tablet study workflow regression passed, including source navigation,
  Arabic material, quizzes, flashcards, guest retention, podcast playback,
  full-screen fitting and theme switching. AI and speech responses are fixtures.
- `npm run test:tablet` passed at desktop (1440×1000), tablet landscape
  (1024×768), tablet portrait (768×1024) and phone (390×844). It exercises actual
  Chromium pen/multi-touch events, native
  canvas output, saved history and native Web Audio. It includes normal Undo,
  Undo while the canvas has zero bounds, redo, one/two-finger PDF and slide
  navigation, scrolling in both directions, finger drawing, dark controls and
  retained page ink. Classification responses are fixtures. Native audio-thread
  peak measurements were 0.206–0.209, below clipping and above the old envelope.

Physical iPad/Apple Pencil hardware and mobile Safari are unavailable in this
cloud runtime. No provider credentials are configured here, so live Groq/Gemini
availability and the deployed account's remaining quota were not verified.
