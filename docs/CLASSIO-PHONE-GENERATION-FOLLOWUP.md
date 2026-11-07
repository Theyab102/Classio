# Phone layout and generation follow-up

This continues commit `bf2f859` without resetting the workspace. It addresses the later phone screenshots and generation feedback. Built-in character customization, original documents, study tools and previously saved avatar assets are retained.

## Changed behavior

- Slide previews, titles and actions use an intrinsic grid. On narrow containers, the complete action toolbar wraps below the preview and title. Library cards use the available phone width. Skin, eye and ink swatches have equal fixed dimensions.
- The assistant, character editor and modal dialogs follow the visual viewport when the keyboard opens. The composer is bounded and scrollable; decorative welcome content yields space while typing. Mobile text fields use a 16px font to avoid focus zoom. Pickers and choice menus track viewport changes.
- Presentation **Topic** and **Design & style** are wrapping textareas next to the main Generate action. The style request travels in that same generation request. One generated style applies immediately; two or three alternatives are optional and come from the model. The preset gallery remains available for manual selection.
- Designs support validated typography, palettes, corners and real checkerboard/grid/dotted/striped/ruled backgrounds. Native PowerPoint patterns and fonts follow the preview. Applying a new style to existing slides keeps their content. Invalid/failed generation preserves the current deck and reports the error; cancellation aborts the active request.
- PDF and PowerPoint library cards show actual file previews. Notes and presentation generation use a catalog of actual source figures. Relevant file figures can be inserted even if notes omit image markers. PDF figures are original page renders; PowerPoint figures are embedded images. File access remains local to the current device. Presentation web fallback is opt-in. Finding more images keeps existing images and prefers the original file.
- Annotation drawing retains per-page/per-slide stroke history, toolbar undo/redo, Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z and Ctrl/Cmd+Y. New edits discard redo. Ordinary writing keeps its own undo history.

## Avatar creation boundary

The sketch creator and community gallery are withdrawn from the character UI under the user's instruction to remove the feature if it cannot create actual requested new hair and clothing shapes. The earlier pipeline generated textures for existing parameterized meshes. It did not implement arbitrary sketch-to-new-3D geometry, and is not presented as doing so.

Face/Hair/Outfit/Details, custom colors, presets, rotation, staged changes, undo/redo, save/cancel, existing worn designs and the private saved catalog remain. Saved designs can be worn, downloaded or removed. Prototype source and adapters remain in the repository; no user assets were deleted. Reintroducing the creator requires a real geometry-generation/fitting pipeline and live fidelity/performance validation.

## Reusable skills

Installed outside Classio at user scope: Vercel's [`ai-sdk`](https://github.com/vercel/ai/tree/main/skills/use-ai-sdk) and Anthropic's [`frontend-design`](https://github.com/anthropics/skills/tree/main/skills/frontend-design). The official OpenAI installer was used with pinned revisions after reviewing the selected files. Installed files have standard `SKILL.md` metadata, no bundled executables/hooks, and matching source hashes. Codex `skills/list` reports both enabled with user scope in two independent project roots, without loader errors. They are reusable within this Codex user home; this does not synchronize them onto other computers or fresh cloud environments. The AI skill is specifically for Vercel AI SDK workflows, including Groq/Google providers. Classio retains its existing REST integration rather than adding an unnecessary SDK migration.

## Verification

| Check | Result |
| --- | --- |
| Production build | Passed; existing legacy lint warnings remain |
| Frontend regressions | 27 suites, 101 tests passed |
| Server regressions | 35 tests passed |
| Desktop and phone browser flows | Passed at 1440px and 390px; 76 screenshots, six real PPTX downloads |
| Individual control bounds | Passed for slide actions, avatar swatches, assistant and character keyboard actions |
| Drawing | Mouse and trusted touch input, per-stroke undo/redo, Ctrl+Z and Ctrl+Y passed |
| File sources | Real PDF/PPTX thumbnails, short slide headings, source catalog, inserted source figures passed |
| Generated designs | Main prompt, wrapping text, real checkerboard, different dotted alternatives, content-preserving restyle passed |
| Final keyboard matrix | 320/768/1024px in both themes; all assistant action buttons remain in the simulated visible viewport |
| Responsive and zoom | 320/768/1024/1280px in both themes, reduced motion, 125% and 150% zoom emulation passed |
| Runtime errors | None captured in passing browser suites |
| Bundle scan | No secret-pattern matches or unrecognized Google keys; existing public Firebase key preserved |
| Whitespace | `git diff --check` passed |

Session artifacts: `/tmp/classio-phone-fixes-final`, `/tmp/classio-tablet-keyboard-final`, `/tmp/classio-phone-browser.log`, `/tmp/classio-tablet-browser.log`, `/tmp/classio-phone-tests.log`, `/tmp/classio-phone-server.log`, `/tmp/classio-phone-build.log`. They last only for this execution workspace. The final targeted keyboard suite ran after the tablet adjustment; the complete desktop/phone suite covers the source and generation changes.

Browser model, image and speech transports are explicit fixtures; actual browser rendering, documents, audio bytes, source images and exports are exercised. These checks do not establish live Groq/Gemini quality, production credentials or quota availability. WebKit installation was blocked by the execution environment's network policy, so physical iPhone Safari remains unverified. Keyboard viewport changes are simulated in Chromium and individual child-control bounds are asserted; a page-width-only assertion previously missed clipped controls.
