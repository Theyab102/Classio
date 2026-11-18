# Page-turn animation

Source PDF and PowerPoint pages now follow horizontal finger movement. The
current page moves toward the drag; the actual adjacent page enters from the
opposite edge with the same displacement. Releasing a completed swipe slides
both pages together over 280 ms. Short swipes and first/last-page boundaries
return to their starting positions over 180 ms. Navigation buttons, keyboard
arrows, page-number fields and search results use the matching direction.

The outgoing page keeps its rendered content and saved annotations while the
next page loads. PDF previews use the original PDF renderer; PowerPoint previews
reuse the existing source slide layout. Ink previews use the same normalized
stroke painter. The preview cache stays around the current page, rather than
rendering an entire document. The loading status does not shift the PDF viewport
while a turn is in progress.

Temporary copies are clipped to the document viewport, hidden from assistive
technology, inert and excluded from ink controls. Pen input waits while pages
move, so a stroke cannot land on a different page than the one being displayed.
Vertical scrolling, zoom panning and explicit finger drawing keep their existing
behavior. Both one- and two-finger navigation use the animation.

Reduced-motion mode switches pages without moving copies. Resizing, zooming,
leaving the file or interrupting a turn clears obsolete copies and animations.
Late preview or animation completion cannot replace a newer turn. Slow PDF
rendering retains the outgoing snapshot until real content is ready, with a
bounded cleanup timer and error cleanup.

Validation:

- Production build passed, with the existing unrelated warnings.
- Frontend: 45 suites and 205 tests passed. New checks cover direction, adjacent
  content, cancellation, boundaries, delayed readiness, interrupted turns,
  resizing, cleanup, reduced motion and pen input during motion.
- `npm run test:tablet` passed at 1440×1000, 1024×768, 768×1024 and 390×844.
  Real Chromium touch events verify finger tracking, both animation planes,
  incoming content, short and boundary rollback, reduced motion, scrolling,
  drawing, undo/redo and retained annotations. Native audio feedback checks also
  pass. Classification responses are fixtures.
- The tablet study workflow regression passed, including keyboard/button/source
  navigation, full-screen fitting, guest retention and the other study tools.
- Drag frames were visually inspected for PDF and PowerPoint content, including
  a large tablet swipe with the next slide visible beside the outgoing page.

Physical Apple Pencil hardware and mobile Safari were not available for testing.
