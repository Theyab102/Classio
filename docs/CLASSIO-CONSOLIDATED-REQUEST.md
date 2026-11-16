# Classio — presentation redesign, product polish, reliability and security

## Working assumptions

- Continue the existing React/Firebase/Vercel application and Piper service. Preserve existing data, records, integrations, branding, document annotations and useful features. Do not rebuild Classio or migrate its working REST integrations to an SDK.
- Classio stays free, with optional Google sign-in and guest access after the existing 13+ eligibility/terms check. Do not add subscriptions, checkout, OTP, SMS or unrelated services to satisfy an audit checklist.
- The checkout starts at `5077ce152d729838502e33551d8481886bd43e49`. The previous environment's uncommitted redesign is described by the supplied handoff, but is absent here. New implementation is reconstruction, not recovered source.
- Keep credentials server-side. Live production verification needs actual access; fixture tests establish integration behavior only. Do not claim universal security, legal compliance, or physical-device verification.

## Original 12-part development request

**CLASSIO — COMPLETE POWERPOINT CREATOR REDESIGN, AI INTEGRATION, LIVE PREVIEW & BUG FIXES**

1. **Repair background removal.** Inspect remove.bg authentication, endpoint, payload, image formats, actual bytes, response validation, timeouts and failure modes. Support PNG/JPEG/WebP; explain animated GIF limitations and handle static GIF conversion when supported. Validate size. Show progress and original/processed comparison, Apply, Cancel and Restore Original; preserve transparency and originals.
2. **Redesign design options.** Use meaningful visual cards, miniature previews, selection states and consistent spacing, typography, borders, corners and transitions.
3. **Remove duplicate layout selection.** Preserve Lecture, Editorial and Visual with clear visual selection; remove the redundant dropdown.
4. **Redesign Add Slide Type.** Offer Title, Content, Image, Comparison, Timeline and Summary cards, retaining Table and Quote. Generated decks should use appropriate types.
5. **Real presentation preview.** Render every slide's actual content/images/styles. Provide Previous/Next, slide numbers, selection filmstrip and full-screen presentation. Allow edit/layout/image/add/delete/duplicate/reorder/theme actions. Updates must be immediate. Share the data model and measurements with PPTX export.
6. **Relevant automatic images.** Prefer relevant uploaded-file figures, then configured search/generation when allowed. Avoid invented URLs, irrelevant decoration, broken results and unnecessary duplicates. Preserve text-only choices and existing images. Include remove/replace/upload/search/generate, size and placement controls.
7. **Improve themes.** Use visual cards with one clear active deck theme; change it without losing content, update the preview immediately and preserve contrast.
8. **Creation workflow.** Describe topic → choose theme/count/style/preferences → generate content/layouts/figures → preview/edit → download. Include progress, cancellation, useful errors and preservation of existing content on failure. Topic/style fields wrap.
9. **Visual quality.** Clean hierarchy, consistent icons/controls, responsive spacing, readable light/dark surfaces, clear navigation and useful loading/error/success feedback.
10. **Verify workflows.** Generation, file figures/search, AI images, cutouts, editing, themes/layouts, CRUD/reordering, previews and real PPTX downloads/rendering. Cover desktop/tablet/phone and both themes. Separate fixtures from live service tests.
11. **Individual slide editor.** Two-panel tools plus large live preview, adapting to smaller screens. Group Text/Content, Images/Media, Layout/Design, AI Tools and Background/Styling. Header: slide number/type/title, movement, duplicate, undo/redo, delete, close. Stage edits; confirm discards; retain drafts on failure; distinguish workspace application from confirmed cloud persistence. Support image previews (never raw base64 fields), replacement, search/upload/generation, cutout preview, restore, resize, crop, aspect-ratio lock and visual placement choices (Automatic/Left/Right/Center/Full Slide/Background). Where practical, direct text editing and image selection/drag/resize/keyboard adjustments update the model. Keep native text undo.
12. **Secure service setup.** Use actual remove.bg at `https://api.remove.bg/v1.0/removebg` with server-only `REMOVE_BG_API_KEY`. Handle auth/credit/rate/format/processing/timeouts distinctly. Verify real transparent output, editor restoration and PPTX preservation when credentials are available. Never invent keys or ask for them in chat.

## Additional product design requirements

Apply these principles across the affected product, with priority on the presentation workspace and initial loading:

- Remove filler badges, redundant labels, random header italics and decorative embellishments. Use monospace only for code or information that benefits from it.
- Use the existing design system consistently: blue accent, spacing scale, typography, radii, controls and stroke icons. Show actual product content; enlarge the main feature and primary preview. Group using space before adding boxes.
- Write specific headings and a clear primary CTA. Wrap/balance headlines, avoid stranded words, keep readable line lengths, adjust kerning/tracking/leading and align elements optically. Use tabular numbers for counters, durations and progress.
- Nest rounded surfaces coherently: inner radius is approximately outer radius minus padding, bounded at zero. Improve text/control contrast, selected states, touch targets and keyboard affordances.
- Make loading visible with stable skeleton layouts and accessible status messages, including before the JavaScript application loads. Investigate the reported repeated reload/flash on iPad. Prevent unnecessary loading-state transitions and layout shifts.
- Design deliberately for phone, iPad/tablet and laptop/desktop: touch, keyboard, pointer, safe areas, on-screen keyboard, portrait/landscape and zoom. Do not shrink a desktop-only layout and call it mobile support.
- Do not add decorative textures, blueprint grids, glass distortion, gradients, grain, ripples, pixel art, photos or other backgrounds merely because a video mentions them. Preserve learner-requested presentation designs; quiet default product surfaces take priority.
- Keep dark-mode controls understandable, accessible and responsive. Cinematic sun/city/knight/glass examples are inspiration for meaningful interaction, not literal Classio features.

## Motion and interaction

- Animate feedback that explains an action/state rather than decorating idle content. Respect reduced motion and cancellation; avoid animation that blocks input or hides an error.
- Prefer ease-out entrances and ease-in-out movements. Dropdown/popover motion about 150–250 ms, with exits around 20% shorter. Start popovers around scale .95 and originate them at their trigger. Everyday transitions should not bounce.
- Use restrained button press feedback (about .97 scale), brief success confirmation, and optional ~30 ms list staggering where it aids comprehension. Preserve focus and scroll positions.
- Consider staging, anticipation, follow-through, arcs, secondary action, snappy timing and shared-object continuity only where they clarify a real action. No forced knob stretching, rubber-band dragging, blurs or springs. Keep transitions interruptible.
- Apply Fitts's law (usable hit areas), Hick's law (clear choices/progressive disclosure), grouping/chunking, familiar conventions and prompt feedback. Remain strict when validating untrusted input.
- Do not install a guessed “Stage Show” skill from a social-video reference. Verify an exact authoritative source before installing executable or reusable agent content.

## Security audit and applicable fixes

Use reproducible tests for applicable defects. Audit authentication/authorization, public assets, dependencies, storage, request validation, origin handling, output sanitization, quotas and deployment headers. Specifically inspect:

- Header-based auth bypass; client-trusted authorization; exposed `.git`, env/credential files and production source maps.
- Public/overbroad Firebase storage/database permissions and cross-account writes. Preserve existing records and do not deploy guessed live rules.
- Client-controlled pricing, repeated coupon redemption, SMS abuse, password stuffing, email pre-registration and account enumeration: classify as not applicable where Classio has no such feature; do not add one.
- Compromised external scripts, leaked server secrets/JWT signing keys, prompt injection, forgotten subdomains, SSRF/cloud metadata access, path traversal, hostile reset-link hosts, clickjacking and open redirects.
- Quota bypass/resource exhaustion, oversized or malformed image/audio requests, unsafe redirects, ZIP/document handling and misleading provider errors.
- Preserve verified identity, server-side secrets, bounded requests, proper TLS/checksums and honest errors. Never treat a browser header or prompt as a security boundary. AI cannot authorize account actions or expose server credentials.

## Privacy, safety and legal hygiene

- Treat social-video fine amounts as unverified, jurisdiction-dependent claims. Do not present them as legal conclusions.
- Keep accurate privacy/terms/copyright pages describing actual collection, local/cloud storage, AI use and named providers. Clearly describe deletion/retention limits; do not promise removal from systems the app cannot control.
- Keep fonts local and analytics/session replay absent unless explicitly authorized. Avoid fabricated reviews/testimonials.
- Preserve age/terms screening; do not describe self-declared age as verified identity or guaranteed COPPA compliance. Do not silently enable Gemini for the app's audience.
- Add appropriate, supportive handling for self-harm requests without pretending Classio provides emergency or clinical care.
- Keep operator/contact/rights-agent configuration truthful. User uploads require rights/permission notices. Registration, production retention settings and jurisdictional decisions remain operator actions, not fabricated configuration.
- Subscription cancellation, trials, renewal notices and payment disclosures are not applicable to this free app. Marketing-email requirements apply only if marketing email is actually introduced.

## Engineering and acceptance

- State working assumptions before implementation. Ask only when missing information prevents a safe, useful decision. Use the minimum coherent implementation; touch relevant code and preserve unfamiliar working features.
- Reproduce bugs with meaningful tests before fixing them; do not weaken assertions or substitute trivial tests. Keep existing frontend/provider suites and extend them for new behavior.
- Inspect the actual UI at representative widths in both themes; test keyboard/touch, long content, loading/failure/success, cancellation and guest/account data preservation.
- Export actual PPTX artifacts and inspect native text, figures, equations, crop, transparency and geometry. Render them in an available office engine; distinguish that from physical PowerPoint/Safari validation.
- Report implemented, tested locally, committed/pushed, deployed and live-provider-verified separately. No production claims based on fixtures. Preserve a clear audit of applicable fixes, non-applicable items and remaining external actions.
