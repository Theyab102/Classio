# Background verification, village characters and study actions

6 November 2026. This pass continues the existing application on
`feat/background-verification-and-villagers`, with speech dependency changes on
`feat/podcast-dependency-security`. It includes the earlier integrated-workspace
branches. Production branches/data have not been changed.

## Verification and the supplied iPhone screenshots

Confirmed source problems:

- The initial `/api/access` request replaced the entire page with “Checking your
  saved verification…”. That screen has been removed from application code.
- A failed request became a generic retry page that masked malformed responses
  and configuration failures. Automatic Firebase restoration could wait indefinitely.
- The browser trusted a successful POST without confirming that its signed cookie
  was retained. The account-link helper treated any HTTP 200 as a successful link,
  even a `stage:new` response, and account linking delayed workspace loading.
- Dark-mode error text and the secondary Google sign-in button had poor contrast.
- Older browsers without `crypto.randomUUID` produced IDs rejected by the course API.

The welcome page now renders immediately while access is checked in the background.
There is no full-screen verification checking message. The date-of-birth form is
shown only after a trusted `new` decision and a deliberate Get started action.
Returning accepted decisions open the workspace. A branded skeleton covers lazy
workspace code loading without a verification message.

Requests and automatic account restoration are bounded. A transient initial failure
gets one delayed retry; returning online or to a failed tab triggers recovery.
Retries preserve the prior decision and never treat a network error as `new`.
Malformed/static HTML responses are rejected. Screening and terms are confirmed by
a follow-up cookie read; discarded cookies produce a first-party-cookie recovery
message with the entered date retained. Cookies remain signed, HttpOnly, SameSite
and persistent; browser storage flags do not grant access.

Guest screening no longer loads Firebase Admin and does not depend on AI credentials.
It still requires a stable server signing secret: `CLASSIO_SESSION_SECRET` or the
existing `AI_RATE_LIMIT_SECRET`, at least 32 characters. The GET route now reports
missing signing configuration explicitly rather than pretending to see a new visitor.
Logs contain a setting/category, never a value or date of birth.

Account linking runs separately from workspace loading, has a timeout, checks the
returned accepted stage, retries twice and resumes on connectivity recovery. Work
is cancelled when the account changes. Existing Firebase sessions can restore
without waiting for an unrelated redirect-result failure. This remains self-declared
eligibility, not identity or documentary proof of age.

**The exact live failure in the screenshots is not confirmed.** This cloud session
cannot reach the production Vercel domain and has no deployment access or securely
bound service credentials. The fixes were exercised locally with the real access
handler. They have not been deployed to `classio-eta.vercel.app` by this task.

## Village character style

The new default style has rounded proportions, a large expressive face, a small
body, soft shading and local SVG artwork. Human, cat, bear, rabbit, fox and deer
characters are available, alongside the existing classic-console and expressive
styles. Natural skin colors, fur colors, hairstyles/animal tufts, eyes, nose, mouth,
face shapes, clothing, glasses and jewelry are customizable live. Inapplicable human
skin/facial-hair controls are hidden for animals. Existing saved attributes and
unknown legacy settings are retained. The art is original Classio work; no Nintendo
characters, branding, game code or asset files are included.

The entry/eligibility flow and policy navigation now support English and Arabic,
including language selection before sign-in, RTL, readable dark-mode controls and
accessible errors. The broader app still has legacy English help, games, group
copy and legal prose; this pass does not claim complete translation of every screen.

## Actual Assistant study actions

The Assistant's **Create from a file** section supports natural-language requests
for notes, flashcards, a practice quiz or a course. It sends only the selected owned
file's notes or explicitly extracted source when the learner submits a request.
The shared Groq/Gemini service proposes an allowlisted action. The proposal is shown
for review and requires **Confirm and create** before execution.

The trusted backend revalidates the arguments, authenticated UID, source and folder.
It creates a **separate file**; it does not overwrite the original. Quiz answers are
interactive and follow normal workspace progress sync. Flashcards and courses open
the existing study tools. A saved result is acknowledged only after a transaction
commits. Retries reuse the same job/result without another provider call or quota
reservation. A source changed during generation causes recovery rather than a stale
save. Completed actions do not reset a new conversation or overwrite newer messages.

The model cannot select another user's paths, fetch arbitrary URLs, execute commands,
delete files or run database queries. Source content is untrusted input. Unsupported
or destructive proposals are rejected. Tool planning and execution use existing
provider preferences, trusted adult-session Gemini gating, quotas and consented
bounded fallback. Provider identity is shown after planning/execution.

These are authenticated account tools. Guests retain the existing direct study
workflows. Model-planned execution is not unrestricted autonomous access, and there
is no automatic paid-provider activation. Real provider/account execution still
needs secure configuration and live validation.

## Dependency security

Compatible overrides update SVG/build helpers, URL resolution and webpack's
development server/middleware. `scripts/dev-host.cjs` adapts CRA's older hooks to
the patched server while retaining the React/CRA application, HTTPS options and
local host checks. The frozen install, production build and ordinary startup were
tested. An untrusted Host/Origin received HTTP 403.

The npm production/dependency audit now reports **30 findings: 0 critical, 0 high,
12 moderate and 18 low**, compared with seven high findings before this pass.
Because CRA/build helpers are dependencies, that audit includes the development
toolchain. It is not a claim of zero dependency risk.

The speech service updates FastAPI, Starlette and Pydantic, uses Python 3.11.17,
and includes hash-locked production/development dependencies. Its executed
`pip-audit` reports **no known vulnerabilities** in the installed project dependency
set. The build installs the production lock with hash verification. Render's Python
pin matches the tested runtime; the host must support that version. No host/billing
resource was provisioned. See the speech README and lockfiles.

## Deployment and diagnostics

1. Review and deploy the new app branch, including `/api` and `/server`, rather than
   uploading only static assets. Set a stable 32+ character signing secret in Vercel's
   server environment and redeploy. Reuse a valid existing quota secret if present;
   a separate session key is recommended. Do not use a frontend-prefixed variable.
2. From an authorized machine run:
   `node scripts/verify-deployment.cjs https://classio-eta.vercel.app`.
   It performs read-only access/AI-configuration metadata checks. It does not claim
   live generation. `--verification-only` instead exercises a fresh fictional adult
   guest screening/terms/cookie flow, without an account, upload or AI job.
3. Secure Firebase Admin, actual Google authorized domains/redirect helpers, exact
   private Storage bucket/CORS and reviewed rules remain necessary for account
   verification, tool transactions and same-account phone/iPad file recall. Preserve
   existing study-group rules and data. See `INTEGRATED-WORKSPACE.md` for specifics.
4. Secure Groq/Gemini keys remain server-only. The key pasted in chat must be revoked
   and replaced securely. Gemini needs current model/free-tier/audience/billing
   review before opting in; no provider was verified live in this environment.
5. Deploy the speech branch using its locked dependencies, persistent private job
   directory and the existing shared signing-secret protocol. Test real licensed
   voices before release. Synthetic QA audio is never a production entry point.

The source fixes do not configure an external Vercel/Firebase account automatically.
No production merge, deployment, rule change, secret rotation or billing action was
performed. Operator/contact details and jurisdiction-specific legal review still
need completion. Local fonts/privacy controls do not guarantee immunity from lawsuits.

## Executed validation

Final results are in `VERIFICATION-VALIDATION.md`, with current logs and screenshots
in the delivered review archive. The earlier integrated handoff/results remain as
historical records and are superseded by this pass where stated here.
