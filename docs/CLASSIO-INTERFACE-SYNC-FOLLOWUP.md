# Interface, tables, AI errors and cloud sync

Continuation from `c978efd374caeccf53dd3752c80c581155e2c5f1` in the existing
Classio repository. The speech repository requires no source changes.

## Interface behavior

- Owner troubleshooting fields moved from the workspace banner into **Settings
  → Account sync → Sync troubleshooting**. A compact status and Retry remain
  visible when saving fails. Browser-storage failures still tell the learner to
  keep the tab open. The project ID, operation and redacted document pattern in
  the screenshot are configuration/diagnostic information, not credentials or
  document contents; ordinary learners do not need that technical detail.
- The most recently opened material has a **Continue studying** cue within
  **Your materials**. The separate recent-work section is removed.
- Folders are taller with larger icons. Document previews have inset frames,
  padding and rounded corners in both layouts. Native slide proportions remain
  intact. Voice/Podcast tabs have equal sizes and centered text. Presentation
  design controls are vertically centered without the clipped left icon.
- Buttons animate on press. Theme changes use the browser View Transition API
  with a fade fallback; repeated choices keep the latest preference. Reduced
  motion skips the animation. Pointer clicks leave no extra outline on dismiss
  and search-clear controls; keyboard focus stays visible.
- DM Sans, Fraunces, Caveat and KaTeX remain packaged local assets. No Google
  Fonts stylesheet or font-host request is needed.

## Table editing

**Table** immediately inserts a 3×3 body-cell table after the complete current
block. **Note options → Custom table** retains row/column/header choices.
Click a cell, choose **Select table**, then press **Backspace** or **Delete** to
remove the whole table. Undo restores it without removing surrounding writing.
Existing text deletion, formatting, merge/split, resizing and mobile stacking
continue to work.

## AI error handling

Truncation is never returned as a complete answer. A truncated response can use
the existing bounded alternate-model/provider route within the same quota
reservation and deadline. If no alternate model exists, the original
`OUTPUT_LIMIT` survives so the course generator can apply its bounded shorter
retry. Provider content-filter blocks stop without trying another provider.
Client quota errors include their wait duration and a specific error code;
shared quota failures remain unavailable rather than silently relaxing limits.

This addresses demonstrated error handling. It does not establish the health,
credentials, quota or output quality of the live deployment's AI providers.

## Confirmed Firebase blocker and required operator action

The latest screenshot identifies Firebase project **classio-4378f**, operation
**read manifest**, path **users/{uid}/preferences/workspace-v2**, error
**permission-denied**, and a **google.com** sign-in provider checked through a
**refreshed ID token**. A repeated sign-in is therefore not an adequate repair.
The updated Settings message points to the owner's deployed permissions.
Local data and the previous cloud manifest remain protected by the existing
failed-restore/pre-save checks.

The checked-in [Firestore rules](../firestore.rules) already allow the
Google-authenticated owner to read/write their preferences. They were not made
public or replaced with a weaker path. An authorized project owner must:

1. Open the default Firestore database's deployed Rules in project
   **classio-4378f** and compare them with the checked-in rules. Review/merge the
   private owner rule rather than replacing unknown production collections.
   In particular, `users/{userId}/preferences/{preferenceId}` must permit
   get/create/update/delete only when the authenticated UID matches `userId`
   and the required Google sign-in provider matches. This covers `workspace-v2`,
   its `workspace-v2-part-*` documents, `assistant` and `assistant-part-*`.
2. Publish the reviewed rules in that project. Verify a real owner's workspace
   and assistant read/write/reload on the deployment, then verify another
   account and unauthenticated requests cannot read those documents.
3. For live AI failures, inspect the deployment's `/api/status`, correct the
   named server-only settings in its existing Vercel project if needed,
   redeploy, then test real text/vision/transcription requests. A Git push alone
   does not create credentials or grant provider access.

This environment cannot perform those production checks: the production URL
request is blocked by the outbound proxy (CONNECT 403); its existing Google
application credential binding cannot initialize through the auth library
(missing `client_email`); and no usable authenticated Firebase/Vercel session
or administrative integration is available. No credential values were printed
or committed, and no live rules or deployment settings were changed.

Original uploaded binaries and audio are still device-local. Metadata sync does
not transfer those originals to another device.

## Reproduction

```sh
CI=true npm test -- --watchAll=false --maxWorkers=2
npm run test:server
npm run build
npm run test:polish
npm run test:notes
npm run test:browser
npm run test:workflows
npm run test:presentation
```

Browser suites use explicit provider fixtures. They exercise actual Chromium
layout, editing, native theme transitions, font requests, file handling and
downloads; they do not verify live Firebase or AI access. Physical Safari/iPad
and production deployment activation require separate verification.

Logs are retained under `/workspace/.onboarding/polish-*.log`; screenshots are
under `/workspace/artifacts/classio/minimal-polish/`. Neither test artifacts nor
provider fixtures are shipped in the production app assets.

Current-instance results: **40 frontend suites / 167 tests**, **42 API tests**
and **6 speech regression tests** pass. The four-device layout/font matrix,
four-device notes/table/course matrix, four-device study-workflow matrix and
desktop/tablet/phone presentation suite pass. The presentation suite produces
nine actual PPTX files. Speech synthesis, classification and generation use
explicit fixtures; these results do not imply live provider success.
The final production build passes with the existing legacy lint warnings.
The broad desktop/phone study suite also passes, including real WAV playback,
uploads, responsive checks in both themes and 125%/150% zoom emulation.
