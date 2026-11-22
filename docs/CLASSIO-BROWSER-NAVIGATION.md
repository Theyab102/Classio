# Same-domain menu URLs and browser history

Classio's main views now use the browser's free History API. There is no added domain, paid service, routing dependency or server queue. Menu changes update the address bar without reloading the workspace. Back and Forward restore the selected view, and real navigation links support copy-link, middle-click and modifier-click.

## URLs

| View | Path |
| --- | --- |
| Library | `/library` (`/` opens the same workspace) |
| Study guides | `/guides` |
| Settings | `/settings` |
| Folder | `/folders/:folderId` |
| Original source | `/files/:fileId/source` |
| Notes / guided lessons | `/files/:fileId/notes` |
| Quizzes | `/files/:fileId/quizzes` |
| Flashcards | `/files/:fileId/flashcards` |
| Podcasts | `/files/:fileId/podcasts` |
| Recordings | `/files/:fileId/recordings` |
| Presentations | `/files/:fileId/presentations` |
| Study games | `/files/:fileId/games` |
| Video / link material | `/files/:fileId/link` |
| Existing study group | `/groups/:groupId` |

`?assistant=1` records the assistant panel being open. Editing notes, answering a quiz, changing a count, and other ordinary actions do not create navigation entries. Global transient dialogs close when browser history changes the underlying view. Their buttons remain actions rather than links.

Initial/canonical URL normalization uses `replaceState`; actual menu navigation uses `pushState`. Repeated metadata updates do not add entries. History stores only a navigation marker and scroll positions. The URL uses encoded IDs, not document names, notes, chat content, credentials, tokens or full file objects. Browser titles describe the view. Current workspace data supplies the folder/file objects, so history cannot restore a deleted or previous account's cached file object.

The existing workspace remains mounted during history navigation. Its saved notes, quiz answers, chats and account boundaries are retained. The prior background-podcast manager continues working while its panel is absent. Explicit source/blank-note/video defaults are preserved, including the legacy guided-lessons default.

## Direct loading and privacy

Vercel rewrites only the named SPA routes to `index.html`. Existing API functions, static assets, private-file denials and legal pages retain their routes. There is no blanket fallback that converts API or missing asset requests into HTML. Browser test servers share a helper that matches these declared rewrites.

Direct file/folder URLs wait for normal authentication/workspace restoration. Missing IDs, deleted material, unknown tools and files outside the current workspace show an unavailable page with a library link. A URL does not upload/share a file or grant access. Guests still need the browser tab where their workspace is stored; signed-in users retain the app's existing sync and device-local upload behavior. Group access still uses existing authentication and database rules.

## Additional reviewed skills

The owner requested more design skills. Two official Vercel skills were added as text-only development instructions:

- `web-design-guidelines`: semantic links, URL state, focus, keyboard/touch access and responsive layout. Its bundled guidelines snapshot and declared SHA-256 were verified. Rules are read from the pinned local snapshot, not fetched remotely during review.
- `vercel-composition-patterns`: separating navigation state from UI rendering and designing reusable component APIs.

The official repository had 34,725 stars when checked on 22 November 2026. Its pinned commit and each imported file's SHA-256 are recorded in `.agents/skills/provenance.json`. Contents were reviewed/scanned and no installer, executable, hook, dependency or binary was imported. Popularity supports provenance; it is not a malware guarantee. The original two skills remain installed.

## Validation

Unit checks cover path round trips, encoded/invalid IDs, defaults, real Back/Forward events, assistant state, native modified-click behavior, no duplicate entries, auth restoration/account switching, and non-interactive excerpts inside file links. Configuration checks preserve API/assets/private routing and test the browser fixture server's path guards.

`npm run test:navigation` exercises the built app in desktop/phone Chromium: menu and file links, Back/Forward, immediate note edits, quiz answers, audio submenus, refreshing/copying deep URLs, missing/deleted files, another guest workspace, transient-menu dismissal, legal routes and API/asset isolation. It uses explicit provider fixtures. The podcast background/voice and desktop workflow regressions cover the existing editor, source files, generation and audio behavior.

The navigation suite is included in GitHub CI. Build/fixture results do not establish that the new Vercel deployment is Ready; production navigation needs its own deployed check.
