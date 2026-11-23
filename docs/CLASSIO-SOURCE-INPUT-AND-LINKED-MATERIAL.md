# Source navigation, writing pads and linked study material

The app's top Back button now leaves any file tab for its containing folder in one click, then returns to Library on the next click. Files stored directly in Library return there immediately. Browser Back/Forward still follow URL history independently.

PDF and PowerPoint source viewers share Left/Right keyboard navigation. It works after choosing Source in the sidebar and after annotating. Search fields, note editors, assistant controls, sliders, menus, modal dialogs and modified shortcuts retain their own keys.

## Huion / Windows / Chrome

Choose Pen, Highlighter or Eraser in the source annotation toolbar. Native pen and left-button mouse-emulated tablet input draw with the selected tool. Hover does not write. Coalesced pointer samples retain handwriting detail, the final lift point is saved, and Undo/Redo use the current page's saved strokes.

For writing-pad drivers that report contact as touch, enable **Drawing tablet mode** under Drawing options. This preference is retained on the device. In that mode, contact on the annotation canvas writes instead of scrolling/swiping the page; select Scroll to navigate. A second touch cancels provisional touch ink. Ordinary finger navigation remains the default outside that mode.

Browser automation exercises native Chromium pen, mouse and touch input, stored strokes, visible canvas pixels and Undo/Redo. It cannot establish physical Huion driver or hardware compatibility; that requires a check with the owner's device. No driver installation or operating-system setting is changed.

## Notes prompts and source evidence

The notes generator's former Topic field is now **Prompt**. A learner can request “Make this easy for Grade 5 students; explain each answer step by step.” Source filenames are supplied separately from the learner's instructions. Notes generation reads available source text, uses explicit grade/year clues together with the actual content, and prioritizes an explicit learner grade request over an inferred level. A filename never substitutes for unavailable source contents.

Prompts remain in checkpoint inputs and every section request for long notes. Generated notes are appended using the existing editor workflow, and the previous writing and recovery checkpoints are retained. PDF extraction keeps page markers for source figures.

## Linked files

Every file view has a visible **Link files** / linked-source control and named links to its companions. The picker includes files across the current workspace's folders. Library cards display linked-source counts. Links are saved in both directions in one workspace update; old one-way links are recognized from either file. Unlinking removes both directions without deleting notes, quizzes, originals or the companion itself.

The shared reader supplies current and directly linked files to notes, guided lessons, quizzes, flashcards, games, podcasts, presentations and the contextual assistant. It reads each actual original where available, falls back to explicitly labelled saved notes/transcripts, and discloses missing originals. Revision questions and answer-key entries can be cross-referenced for explanations. No unrelated file, transitive link, missing ID or another account's workspace is read.

Context remains bounded. Each linked source receives part of the available text budget so a long revision pack cannot crowd out its answer key. Long note/course/presentation sections retain a bounded cross-reference to the companion sources. These are source excerpts rather than an assertion that every page of an arbitrarily large upload was sent in a single request. Existing local-only originals, cloud sync, provider quotas and free-only routing remain in place.

## Validation

- Unit tests: reciprocal/unlink behavior, old links, account boundaries, unavailable sources, context budgeting, long-note prompt retention, keyboard ownership and tablet input.
- `npm run test:source-input`: real TXT/PDF/PPTX uploads and file reads; cross-folder links; actual note/quiz/assistant request bodies; Grade 5 prompt; unlink/refresh retention; hierarchical Back; PDF/PPT keyboard page flips; Chromium pen/mouse/touch ink, pixels and Undo/Redo; desktop/phone layout.
- Existing navigation, tablet, background podcast and authentication regressions verify integration. Provider responses in browser tests are fixtures, not live AI inference.
- The new source/input browser suite is included in GitHub CI alongside build, unit tests, dependency audit and credential protection.
