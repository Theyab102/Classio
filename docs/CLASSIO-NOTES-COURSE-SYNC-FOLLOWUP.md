# Notes, tables, course generation and sync follow-up

Continuation from `cb65b002925c1b21e53ef893ba4326f1bcc3562c`, implementing the
latest unresolved issues in the project handoff. The speech repository remains
at `1dff2b42f6e424374e6d1c84421afb49bd9497b7`; it needs no source change for this round.

## Editing and controls

Reproduced the empty-note placeholder in every empty table cell and nested
tables from repeated insertion. The placeholder now applies only to the empty
editor root. Table insertion goes after the complete current block or containing
table, preserving words/headings. A visible Table button opens row/column/header
choices, and Make a table exposes the AI action directly. Ordinary tables start
with body cells; headers are an explicit choice.

Table actions include rows, columns, header toggles, deletion, selection-based
merging, splitting and an adjacent-cell merge for touch devices. Column drags
resize neighboring columns within the editor and retain usable minimum widths.
Stored widths fit the current viewport. Older tables with too many columns to
fit switch to vertically stacked rows on narrow screens without rewriting their
data; original headings and merged-cell data remain intact. Wider screens restore
the column view. The editor uses the workspace width, and rows grow with content.
Text marks, fonts, sizes, alignment and column widths survive navigation/restoration.
Images retain the existing aspect-preserving resize implementation.

Course/podcast controls share consistent sizing and bottom alignment. This also
overrides the global button `align-self:center` rule that caused the mismatch.
Topic fields wrap, and the note Topic label remains distinct from its textarea's
value. Podcast tabs expose their selected state. Source slide thumbnails fit the
actual aspect ratio rather than clipping to a fixed-height box. Keyboard focus
indicators remain visible.

## Courses and source handling

Course generation now requests a compact outline followed by individual lessons.
Long sources are partitioned in order so later lessons receive later source
sections. Short sources remain available in full for each distinct focus. Source
input and per-request excerpts remain bounded; this is not a claim that an
arbitrarily large document is sent in full. Arabic/language preferences and the
existing math JSON repairs remain supported. An unreadable source without an
explicit topic gets a clear error rather than a course invented from its filename.

The current course and progress remain active until every replacement lesson
passes validation. Progress names the current step, cancellation rejects late
results, and truncation/malformed output receives a shorter retry. There are at
most two extra requests across the whole generation, and at most one retry per
step: a 12-lesson course uses at most 15 calls including its outline. Normal
server quotas still apply. Rate limits/credential failures are not retried in a
loop. Incomplete courses are not published and oversized lessons are not silently
clipped into apparently successful content.

## Existing and future images

Include relevant images is an accessible pressed-state button. Turning it off
prevents new image insertion, including PDF markers. Automatic web candidates
must match the requested subject and the surrounding section; uncertain matches
are skipped. This remains a conservative text/metadata heuristic, not vision
verification of every figure.

The Images dialog also reviews pictures already saved in notes. It flags the
legacy forest/Geograph example for review, distinguishes newly recorded manual
selections, and lets the learner remove a chosen image with Undo. No automatic
deletion of older or manual pictures occurs. Clear image credits are removed with
their image; unrelated writing and other images remain. New provenance survives
HTML sanitization/restoration, and organizing notes preserves original image nodes.

## Firebase: investigation and remaining operator action

The public application configuration points to `classio-4378f` and its Firebase
auth domain. Source `firestore.rules` already allows the Google-authenticated
owner to read/write their profile, preferences and saved notes. That includes
workspace/assistant manifests and their part documents. These source rules were
not broadened or deployed.

Current access limits prevent a production repair claim:

- The production website request was rejected by the outbound proxy with CONNECT
  HTTP 403.
- The existing `GOOGLE_APPLICATION_CREDENTIALS` binding was checked through the
  Google authentication library for a read-only Rules API inspection. It could
  not initialize authentication (missing `client_email`). No credential values
  were printed, no alternate credentials were invented, and no live rules or
  production account records were read.
- The screenshot proves an access-denied response but does not identify its exact
  document/operation or the deployed rules. Those remain unverified.

Client errors now retain the project, read/write operation and private document
pattern. An expandable notice exposes these diagnostics without account IDs,
tokens, prompts or document contents. Assistant sync failures also stay visible.
Retry refreshes the ID token and reports its sign-in provider, distinguishing it
from merely linked-provider metadata. A denied pre-save manifest read stops
before new parts publish. Local caches and the previous manifest are retained;
editing locally does not replace an access-denied notice with a generic success.

An authorized operator must:

1. Open Firebase project **classio-4378f**, inspect the deployed default Firestore
   rules, and use the new Cloud sync details after the client update to identify
   the denied operation. Confirm the account is using the expected Google sign-in
   provider, not merely a linked Google account.
2. Compare the existing deployed rules with the owner-scoped paths in
   `firestore.rules`. Review/merge the necessary owner path rather than blindly
   replacing unknown production rules or making collections public. Verify:
   `users/{uid}`, `users/{uid}/preferences/workspace-v2`,
   `workspace-v2-part-*`, `assistant`, and `assistant-part-*`.
3. Publish only the reviewed rules in the correct project. Test an actual owner's
   workspace and chat read/write/reload, with a second authenticated browser as
   appropriate. Verify unauthenticated and cross-account access remains denied.

Original uploads and audio remain device-local; cross-device metadata sync is
not a promise to transfer those binary files. Live provider keys and quotas,
physical Safari/iPad behavior, deployment activation and real-account sync are
separate from the fixture-based tests below. No new provider credentials or
production Firebase/Vercel/Hugging Face configuration were applied.

## Reproduction and validation

From the application checkout:

```sh
CI=true npm test -- --watchAll=false --maxWorkers=2
npm run test:server
npm run build
npm run test:notes
npm run test:browser
npm run test:workflows
npm run test:presentation
```

`test:notes` exercises actual Chromium selection, formatting, merging/splitting,
Backspace/Undo, constrained drags, growing rows, saved-width/mark restoration,
wide-table responsive rendering, AI table insertion, image review and the image
toggle. It covers 1440×1000 desktop, 1024×768 landscape tablet, 768×1024 portrait
tablet and 390×844 phone in both themes. Its source-language, truncation, rate-limit
and cancellation transports are explicit fixtures. The existing suites retain
their study, account isolation, upload, drawing, podcast, presentation and actual
PPTX/browser WAV coverage. Jest resolves Tiptap's CommonJS entry points so the
table command tests execute the real editor dependencies.

Run speech regressions from its checkout with
`/workspace/.venvs/classio-tts/bin/python -m unittest -v test_server.py`.
Speech synthesis is mocked in those six tests; no live Piper/provider success is
inferred from them.

Final current-instance results: **39 frontend suites / 161 tests**, **39 API
tests**, **6 speech tests**, and the production build passed. The build retains
the existing legacy lint warnings. All four browser commands above completed
successfully: the new four-device notes matrix; broad desktop/phone study flows
and responsive/zoom checks; four-device study workflows; and desktop/tablet/phone
presentation checks with nine real PPTX downloads. Browser transports are fixtures;
WAV playback/decoding and PPTX creation are real. No fresh independent Office
renderer, physical Safari device, live AI request, real-account Firebase sync or
production deployment verification is implied.

Local artifacts and logs are retained under
`/workspace/artifacts/classio/current-fixes/` and
`/workspace/.onboarding/fix-*.log`. The additional workflow/presentation temporary
artifacts were copied into that stable artifact directory after their successful
runs. No source map, credential, fixture response or test screenshot is added to
the production application assets.
