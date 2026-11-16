# Assistant and study workflow fixes

Follow-up to the screenshots and detailed workflow report supplied on 16 November
2026. Existing notes, uploaded originals and study records remain supported.

## Changes

- The assistant sidebar action toggles open/closed. Search and nested dialogs
  handle their own Escape key without closing the assistant behind them.
- Chats have independent messages and drafts, automatic titles, New chat,
  history selection, rename and deletion. Existing single-chat records migrate.
  Accounts remain isolated; guest chats remain session-only. Saved history keeps
  text, not image attachment bytes, matching the previous privacy behavior.
- A visible panel divider supports pointer/touch resizing and keyboard arrows on
  tablets and desktop screens, including portrait tablets. Enlarge/Restore uses
  the same bounded width. Phones retain a
  full-width composer. A Library question no longer dereferences a missing activity.
- Cloud snapshots use bounded private documents and a final manifest instead of
  putting all notes, base64 images and decks in one Firestore document. Existing
  workspaces remain readable. A successful migration removes the obsolete root
  folders field after the new manifest commits, keeping profile fields. Failed
  parts leave the last manifest intact. Retry
  and reconnection paths keep the local cache; offline deletions survive merging.
  Original uploads and podcast WAV files remain device-local.
- Cancelling note, quiz or card generation aborts the request and prevents a late
  response from changing the saved work.
- AI JSON parsing repairs common single-backslash math and raw line breaks without
  executing output. Truncated/malformed results receive a readable error and keep
  the existing work. Quiz/card requests are batched for custom counts from 1 to 50;
  the editable count starts at zero and can be cleared without being forced to one.
- Source-based learning reads the original file before older generated notes.
  Arabic text produces an explicit Arabic instruction; configured language choices
  are read at request time. Markdown and note editing use automatic text direction.
  Grouped PowerPoint text is included in extraction. Unreadable files do not cause
  filename-based fabricated lessons. Scanned/image-only files still need readable
  text or actual image analysis; this change does not invent an OCR service.
- Guided lessons use source length or a selected count up to 12, retain all checks,
  and can restart the saved course/questions. Course and quiz replacement use app
  dialogs and retain the old work until generation succeeds.
- Settings offer subject-independent learning styles. Visual styles have local
  presets, a general study preview and a clear removal action. Presentation style
  instructions apply during generation, can be cleared, and are not automatically
  restored as a saved prompt. Choosing a standard theme clears custom styling.
- Flashcards use a content-sized card instead of a fixed inner scrolling text
  box. Full screen has a browser mode and an in-app fallback. Very long content
  can require outer-page scrolling; it is not silently clipped or made unreadable.
- Podcast host voice selectors are visible, expose the configured catalog, keep
  selection, and provide an explicit rebuild action. A known six-voice fallback
  is labeled when catalog loading fails. This does not claim an unavailable voice
  service is working. New episodes receive varied characters saved with the episode;
  characters can be shuffled. Transcript/audio/delete actions are proper buttons.
  The recording-only language selector and duplicate heading are hidden in podcast
  mode so its real voice controls and player receive the space.
- The speech service allows a complete episode plus an immediate voice rebuild
  within its per-client limit, while retaining its body/text, concurrency and
  total-instance limits. Rate-limit errors show the actual retry interval.
- The recording empty-state microphone has a valid stroke color and centered
  positioning, eliminating the empty blue square. Search clearing has an explicit
  44px button. The theme switch supports dragging, shows sun/moon, and has a larger
  hit target. Textarea resize grips are removed from routine forms.
- Workspace back controls cover materials, tools and Settings. Source slides have
  fit-to-slide and full-screen controls, arrow-key navigation and horizontal touch
  swipes. Generated-deck preview supports the same swipe direction and full-screen
  fallback. Drawing controls reveal extra options when needed.
- Automatic image matching prefers explanatory diagrams and rejects incidental
  photographs such as the forest/crater example. Manual image selection remains
  available; semantic matching is a conservative heuristic, not visual certainty.

## Validation and boundaries

- Production build passes, with the repository's remaining legacy lint advisories.
- Frontend: **37 suites / 145 tests**. API: **39 tests**. Speech service: **6 tests**.
- The new workflow suite passes at 1440×1000 desktop, 1024×768 touch tablet,
  768×1024 touch tablet and 390×844 phone. It covers chat history/toggling/resizing,
  search preservation, empty-note cleanup, Arabic text and actual PPTX import,
  lesson restart, custom counts, flashcard overflow/editing, voice rebuilds,
  recording icons, theme dragging, full-screen menus and both swipe directions.
  Portrait tablet/phone checks disable the native full-screen API to verify the
  in-app fallback. The suite produces **36 screenshots**.
- The broad study suite passes desktop and phone flows, source imports, actual
  WAV decoding, annotations, existing avatar tools and six real PPTX downloads
  (**88 screenshots**). The final responsive run passes 320/768/1024/1280px in both
  themes, simulated tablet/phone keyboards, reduced motion and 125%/150% zoom.
- The presentation suite passes on desktop, touch tablet and phone, including
  removal of prior custom styling when choosing a standard theme. Its **nine
  real PPTX downloads / 45 slides** pass ZIP checks and independent rendering
  with Aspose.Slides 26.11. Evaluation-renderer PNGs have a watermark; the exported
  PPTX files do not. Its **25 screenshots** include editor/export checks.

Commands: `npm run build`, `CI=true npm test -- --watchAll=false --maxWorkers=2`,
`npm run test:server`, `npm run test:browser`, `npm run test:presentation`, and
`npm run test:workflows`. In the speech-service virtual environment, run
`python -m unittest test_server.py`.

Local review files are in `/workspace/artifacts/classio/workflow-fixes/`. Detailed
artifacts are in `/tmp/classio-workflow-final`, `/tmp/classio-workflow-full-final`,
`/tmp/classio-workflow-responsive`, `/tmp/classio-workflow-decks-final` and
`/tmp/classio-workflow-rendered-decks`. Logs use `/workspace/classio-workflow-*.log`.

Provider responses in browser checks are explicit fixtures; they do not establish
live quotas, content accuracy or provider availability. Physical Safari/iPad,
Microsoft PowerPoint and real-account sync remain separate checks.

The production URL remains blocked by this environment's outbound proxy (CONNECT
403). Firebase production permissions, quotas and deployment status cannot be
verified here. Source rules already allow the account owner's preference records;
if deployed rules differ, an operator must deploy/review them. No production rules
or credentials are changed by this patch.
