# Equation coverage, built-in avatars and the keyboard gap

This continues `a57a5bb` and preserves the existing workspace, saved character indices and saved wearable assets.

## Equation coverage

The remaining raw study-content paths now use the shared renderer: multiple-choice options, correction messages, expanded explanations, game prompts and answers, older tutor/chat views, course headings, podcast headings/menus, material previews, and PowerPoint source text/tables. Existing notes, quizzes, flashcards, lessons, podcast transcripts and presentation previews share the improved parsing. Equation-aware PowerPoint export and speech use the same repaired notation.

The parser handles common keyboard physics notation such as `vy = v0y - g t`, initial-velocity subscripts, Unicode Greek symbols and comparisons, and double-escaped delimiters from stored model output. KaTeX's chemistry extension supports chemical expressions; standard fractions, roots, matrices, aligned expressions, calculus and units remain available. Code examples and original file names remain literal. Unsafe commands stay disabled and invalid expressions request review. This is a shared renderer for supported notation, not arbitrary user-defined LaTeX packages.

Fill in the Blank now keeps the full question rather than removing a word that may contain the formula's closing delimiter. Material previews finish an equation before truncation and protect comparisons from HTML cleanup. Inline paragraph spacing is retained. Source PDF canvases continue showing the original document.

## Built-in customization

The character keeps its original soft proportions and Classio colors. Tops now use shaped fabric geometry with shoulders, an open neckline, separate sleeves, hems, collars, cuffs, pockets, seams and selected knitted/printed details. Shirt silhouettes and sleeve lengths vary by garment.

| Category | Current choices |
| --- | --- |
| Tops | 36, including 20 additions |
| Hair | 33, including 16 additions and a bald option |
| Bottoms | 9, including Match outfit |
| Shoes | 10 |
| Hats | 18, including None |
| Glasses | 13, including None |
| Accessories | 9, including None |
| Eye shapes / expressions / eyebrows | 8 / 10 / 9 |
| Facial hair | 8, including None |
| Necklaces / earrings | 10 / 10, including None |

The wardrobe has Tops, Bottoms, Shoes and Colors categories. Details separates Hats, Glasses, Jewelry, Facial hair and Accessories. Clothing accents, bottoms, shoes and accessories have independent colors. Original indices are unchanged and new choices append to the catalog. Randomizers include the expanded selection and retain the name and saved catalog.

The primary avatar render takes priority over thumbnails. Offscreen thumbnails wait until visible; pending previews show a loading indicator instead of an unrelated default outfit. Obsolete queued work is discarded, materials are disposed, camera framing accommodates tall hats, and solid hats suppress protruding top buns/puffs without changing the saved hairstyle. Previously identical curly/textured hair, full/short beards, drop/hoop earrings and shoe options now have distinct geometry.

The sketch creator remains withdrawn. This update expands functional built-in items; it does not reintroduce arbitrary AI geometry generation.

## Keyboard gap

The assistant's controls follow the visual viewport, while a separate opaque themed surface covers the full drawer area behind them. That surface covers the region exposed by iOS's translucent keyboard accessory strip, which previously showed the document through the gap. Existing drawer dismissal and body-scroll restoration remain available.

## Verification

| Check | Result |
| --- | --- |
| Production build | Passed; existing legacy lint warnings remain |
| Frontend regressions | 29 suites, 114 tests passed |
| Server regressions | 35 tests passed |
| Full desktop/phone browser flow | Passed at 1440px and 390px; 88 screenshots and six PowerPoint downloads |
| Expanded avatar editor | New item selection, rotation, category navigation, save/reopen and cancel passed on desktop and phone |
| Geometry catalog | All tops and each hair, wearable and face option have distinct bounded geometry; saved indices remain compatible |
| Equations | Common physics, escaped delimiters, fractions, roots, calculus, matrices, chemistry and equation-safe excerpts passed |
| Browser equation integration | Math in quiz/game choices, source slides, notes and actual PowerPoint vector exports passed |
| Keyboard gap | Opaque full-height themed drawer backing and visible action bounds passed in viewport emulation |
| Final responsive smoke | 320/768/1024/1280px, both themes, reduced motion, keyboard sizing, 125% and 150% zoom passed |
| Bundle scan | No server secret patterns or unrecognized Google keys; existing public Firebase key preserved |
| Whitespace | `git diff --check` passed |

The full browser flow ran before the final consolidation of the legacy table/rich-text helpers. The final production build, full unit suite and responsive smoke ran after that consolidation. Session artifacts: `/tmp/classio-equations-avatars-final`, `/tmp/classio-equations-avatars-smoke`, `/tmp/classio-equations-avatars-browser.log`, `/tmp/classio-equations-avatars-smoke.log`, `/tmp/classio-equations-avatar-tests.log`, `/tmp/classio-equations-avatar-server.log`, `/tmp/classio-equations-avatar-build.log`. They last only for this execution workspace.

Provider transports in the browser suite are explicit fixtures; actual WebGL rendering, source documents, audio decoding and PowerPoint exports run in the browser. Physical iPhone Safari is unavailable in this environment; visual-viewport changes and the backing surface are checked in Chromium emulation.
