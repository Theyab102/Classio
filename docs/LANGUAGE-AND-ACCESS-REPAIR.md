# Language switching and the live verification error

6 November 2026. Branch: `fix/verification-config-and-language-switch`.
This continues the existing Classio upgrade without changing the database or
the separate speech service. Earlier validation documents describe earlier passes.

## Confirmed findings

The supplied production screenshot shows the exact message returned by
`api/access.js` for `VERIFICATION_CONFIG_MISSING`. In this implementation that
means the deployed function cannot select a signing secret at least 32 characters
long. This is a server configuration failure, not an iPhone date-input problem.
The screenshot establishes the error path; this task has not independently read
the production deployment's configuration.

Some API errors and editor messages were translated before entering React state.
An error created while Arabic was selected stayed Arabic after switching to
English. A new browser regression reproduced that failure against the previous
build. Memoized labels also received a translator whose identity did not change
with the interface language, and the rich-text editor's accessible label could
retain its initial language.

## Source changes

- API errors retain canonical text and their status/code/retry information.
  Entry, notes/chat, Assistant/actions, courses, recording and podcast controls
  translate UI messages when rendered in the current interface language.
- Parameterized message keys keep learner-provided names separate from translated
  UI copy. Changing the interface never reverse-translates notes, uploaded files,
  submitted questions or existing AI answers.
- Locale subscribers receive a translator scoped to the selected language, so
  memoized labels recompute. Editor accessibility labels update without replacing
  the editor, its content or undo history. Recording/course controls subscribe
  directly instead of depending on a parent rerender.
- A missing server key stays a truthful unavailable state, without the automatic
  cold-start retry. Manual and connectivity recovery remain available. Google
  sign-in is disabled while the access service is known to be unconfigured.
  No browser flag grants access and the checking screen remains removed.

Interface language and AI response language are independent settings. Existing
learner content remains in its original language. Legacy screens still contain
English copy; this pass does not claim whole-application translation.

## Required live repair

The source changes cannot supply a private production signing key. In the
existing **Classio** Vercel project, securely set a stable random
`CLASSIO_SESSION_SECRET` of at least 32 characters for **Production**, then
redeploy the current production deployment so its functions receive the setting.
Keep it server-only, unprefixed, out of source control and out of chat. Reuse a
valid existing key; unnecessary rotation invalidates saved browser cookies.

`AI_RATE_LIMIT_SECRET` is used only if `CLASSIO_SESSION_SECRET` is unset. A
nonempty, too-short dedicated value must be corrected, even if the quota key is
valid. Firebase Admin and AI credentials are not required for guest screening;
they are still required for their account and AI features.

Deployment access has been requested through a secure `VERCEL_TOKEN` environment
connection, with optional `VERCEL_PROJECT_ID` and `VERCEL_ORG_ID`. Declaring a
requirement in the environment draft does not bind a credential or activate
network access. At the last observed runtime check those credentials and outbound
Vercel access were unavailable. No production deployment, setting or user data
has been changed by this pass.

After configuration, from an authorized machine run:

```sh
node scripts/verify-deployment.cjs https://classio-eta.vercel.app --verification-only
```

This uses a fresh fictional adult guest: new decision, screening, terms, signed
cookie retention and returning accepted access. It creates no account, upload,
course or AI job. Then verify a returning signed-in account on another device
with the actual Google/Firebase deployment. Do not remove server checks to make
an unconfigured service appear successful.

## Executed validation

- 78 frontend tests in 16 suites passed, including Arabic-to-English API errors,
  parameterized messages, memoized labels, the fail-closed entry screen and
  recovery after correcting configuration.
- 47 backend tests passed, covering signed eligibility, missing-secret failure,
  cookie tampering, age/terms decisions, account ownership, AI routing and tools.
- Error-level ESLint and `git diff --check` passed.
- The production build passed with existing CRA lint/deprecation warnings.
- `test:language` passed against that build: Arabic-created verification/API
  errors and generation messages returned to English; original Arabic writing,
  the live editor instance and the submitted Assistant question were preserved.
- `test:ui` passed, including actual Settings English-to-Arabic-to-English controls
  at 320, 390, 844, 768, 1024 and 1366 pixel viewport widths. These are Chromium
  viewport checks, not physical Safari tests.
- The background-verification browser suite passed actual signed guest eligibility,
  reload/connectivity recovery, discarded-cookie rejection and five phone/tablet
  viewport fits. Its dark entry and village editor Axe samples had no violations.
- The real local API passed `verify-deployment.cjs --verification-only` for
  screening, terms, signed-cookie retention and returning guest access.
- Build-only server credential sentinels and Google Fonts hosts were absent from
  generated files. No Google Fonts, Signum or replay integrations were added.

AI-success browser fixtures are not live-provider tests. Logs and screenshots
accompany the review package. This is not whole-app accessibility certification.

Real Groq/Gemini generation, production Google sign-in, actual cross-device
Firebase file recall and physical iPhone/iPad Safari remain external checks.
No speech code was changed or retested in this repair pass.
