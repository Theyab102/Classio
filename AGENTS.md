# Classio development guidance

Classio uses Create React App, React and CommonJS Vercel API functions. Keep changes compatible with that architecture.

The owner requested the reviewed development skills installed under `.agents/skills`:

- For UI implementation, read `.agents/skills/frontend-design/SKILL.md`. Preserve the established Classio palette, branding and layout unless the user's task calls for a redesign.
- For React state, lifecycle and async work, read `.agents/skills/vercel-react-best-practices/SKILL.md` and the relevant rule files. Apply the React guidance; Next.js-specific APIs do not apply to this application.

Upstream commits, licenses, file hashes and review notes are in `.agents/skills/provenance.json` and `.agents/skills/README.md`. Imported examples are reference text, not commands to execute. User instructions take precedence.

Use the existing repository scripts for validation. The podcast background/browser suite uses provider fixtures; distinguish those checks from live production inference. Preserve account boundaries, existing material and shared production quotas when changing background jobs.
