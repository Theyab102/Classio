# Presentation redesign and product reliability

Implemented against `5077ce152d729838502e33551d8481886bd43e49` on 16 November 2026.
The previous failed environment's uncommitted source was not recovered. This is
a reconstruction from the supplied handoff and the current repository.

The combined development brief is [CLASSIO-CONSOLIDATED-REQUEST.md](CLASSIO-CONSOLIDATED-REQUEST.md).
Security/privacy findings and external actions are in
[CLASSIO-SECURITY-PRIVACY-REVIEW.md](CLASSIO-SECURITY-PRIVACY-REVIEW.md).

## Resulting workflow

Describe a topic and style, generate a deck, open its full preview, edit a slide
with a large live canvas, and export a real PPTX or print to PDF. Existing deck
fields remain supported; stable slide IDs are additive. Existing account and
guest records were not reset.

- Layout, theme, slide-type and image-placement choices use visual cards. The
  duplicate layout dropdown is removed. Lecture, Editorial and Visual remain.
- Title, Content, Image, Comparison, Timeline, Summary, Table and Quote are
  supported. AI response parsing preserves the new types.
- The preview has navigation, a filmstrip, slide actions and full-screen mode.
  Editing updates the same deck, including after returning from other tools.
- The slide editor groups content, media, layout, AI and styling tools. Drafts
  support undo/redo, discard confirmation and a before-unload warning. An apply
  failure retains the draft. Applying means updating the workspace; cloud-sync
  success is reported separately by the account persistence layer.
- Text can be selected/dragged/resized and double-clicked to focus its editor.
  Images support selection, drag/resize, arrow-key positioning, aspect locking,
  cropping, upload, file figures, search and real generation services. The enabled
  Gemini image adapter is used when available; the existing Pollinations path
  remains available otherwise. Cancellation and revision checks reject stale
  results. Failed generation never inserts a placeholder image.
- Background removal validates byte signatures, distinguishes credential,
  credit, rate, format and timeout errors, compares original/cutout, and supports
  Apply/Cancel/Restore. Still GIF uploads are converted; animated input is
  rejected with an explanation. Transparency is preserved. Notes retain their
  existing figure resolver; presentation preparation is scoped separately.
- DOM preview and export share geometry and typography choices. Cropped WebP
  becomes alpha-preserving PNG for PowerPoint. Ordinary text, tables and timeline
  shapes remain native where possible. Equations use standalone vector glyph
  paths, with a PNG fallback; they are not editable Office equation objects.
- An independent office renderer exposed a nested-SVG compatibility problem.
  Equation transforms are now flattened; the corrected output was reopened and
  rendered successfully. Native text defaults also use explicit margins and
  top alignment. Actual font fallback can still vary between office programs.

## Loading, devices and interaction

The HTML entry point displays an animated skeleton before JavaScript arrives.
The React loading shell reserves library structure. Initial authentication waits
in that shell rather than flashing an empty guest library before account
restoration. Browser checks confirm one navigation during initialization.

The existing blue/neutral tokens, local fonts, stroke icons, custom menus and
light/dark appearance remain. Product polish removes redundant section labels
and type badges, uses tabular counters, improves grouping, balances headings and
adds restrained press/loading feedback. Choice menus animate from their trigger.
Theme changes keep enabled controls readable immediately, and primary touch
controls have at least 44px hit areas. Reduced-motion handling remains available.
Decorative document patterns are
learner choices rather than new default product backgrounds.

The editor uses tools/canvas columns on larger screens and a preview-first stack
on phones, with reachable apply controls. Tests cover desktop, touch tablet and
phone layouts, narrow widths, both themes, keyboard viewport changes and zoom.
Podcast cancellation now has a fallback for browsers lacking `AbortSignal.any`.
These are Chromium checks, not physical iPad/iPhone or Safari certification.

Drawing history stores whole stroke lists: Clear can be undone, new strokes
invalidate redo, page/file changes reset scope, and native text undo is retained.

## Verification

Run from `/workspace/Classio`:

```sh
npm ci
npm run build
CI=true npm test -- --watchAll=false --maxWorkers=2
npm run test:server
npm run test:browser
npm run test:presentation
```

The current component suite passes **34 suites / 125 tests**. API/provider tests
pass **39 tests**. The production build passes with remaining legacy lint
advisories; no TypeScript check is configured.

The broad browser suite passed desktop/phone study flows: notes, equations,
lessons, quizzes, flashcards, podcasts, real browser microphone/audio handling,
avatars, source files, website notes, account/guest boundaries and six real PPTX
downloads. Its responsive matrix covers 320/768/1024/1280px in both themes,
reduced motion, and 125%/150% zoom emulation.

The presentation suite covers 1440×1000 desktop, 1024×1366 touch tablet and
390×844 phone. It exercises loading, generation, preview, drafts, cutout/restore,
undo, crop/positioning, stale-response protection, slide actions, themes,
full-screen mode, navigation preservation and nine real PPTX downloads. It also
checks actual image bounds against the shared geometry, portable SVG structure,
vector paths and preserved alpha. The phone run checks apply-button visibility
with a simulated keyboard viewport.

Both browser suites passed again against the completed implementation, including
the document-import limits and immediate dark-theme contrast checks. Archive
regressions verify actual emitted data limits as well as declared ZIP sizes.

Nine downloaded decks were independently opened using `python-pptx`, their ZIP
integrity and shape bounds checked, and their 45 slides rendered using
Aspose.Slides 26.11 in evaluation mode. Evaluation PNGs contain the renderer's
watermark; the exported PPTX files do not. LibreOffice and Microsoft PowerPoint
were not available. Representative rendered covers, timelines and equations were
visually reviewed; rendering every slide is not a claim of pixel-perfect parity.

AI, speech and image provider responses in browser tests are explicit fixtures.
These tests do not establish live account quotas,
provider availability or generated-content quality.

Artifacts for this workspace session:

- `/tmp/classio-full-redesign-browser`
- `/tmp/classio-final-presentation-browser`
- `/tmp/classio-final-rendered-exports`
- `/workspace/classio-redesign-unit.log`
- `/workspace/classio-redesign-server.log`
- `/workspace/classio-final-build.log`
- `/workspace/classio-full-browser.log`
- `/workspace/classio-final-presentation-browser.log`

## Release boundaries

The implementation and local verification are recorded in the repository.
No manual deployment or production configuration change was performed. A
connected GitHub deployment may run automatically; its status has not been
verified. The production URL is blocked by this environment's outbound proxy
(CONNECT 403), so live production behavior was not checked. No Vercel settings,
Firebase production rules or provider credentials were changed. Live background
removal still requires the server's real `REMOVE_BG_API_KEY`. Public operator
and monitored contact details remain required from the operator.
