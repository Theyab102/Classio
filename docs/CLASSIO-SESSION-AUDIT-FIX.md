# Classio session recovery and dependency fixes

18 November 2026. Follow-up to the AI upgrade and screenshots showing “Start a secure Classio session or sign in again” across notes, presentations, flashcards and podcasts.

## What changed

The shared client previously returned a rejected authentication request without renewing its credentials. It now waits for Firebase initialization and refreshes an expired account token once before replaying the same request. A lost or expired guest cookie similarly renews its signed session and CSRF token once. Concurrent refreshes share the renewal. Account changes and cancellation stop late retries. Revoked accounts, forbidden requests and configuration/service errors do not trigger a guest downgrade or repeated generation.

The server previously labeled configuration errors and certificate/network failures as a sign-in problem. Fixed public messages now distinguish expired/revoked credentials, unavailable verification and incomplete server setup. Firebase still validates token signatures, issuer, project, expiration and Google sign-in provider. Production rejects Firebase Auth emulator configuration and a mismatched administrator project. Vercel function logs include fixed diagnostic codes and missing variable names, never credential values or raw Firebase/provider error bodies.

An attempt that completes no outline or sections now shows a compact “Generation hasn’t started yet” retry control. It does not claim that zero sections are saved. Actual outlines and completed sections retain accurate recovery controls and durable resume. The presentation workspace no longer draws a blue outline around the entire pane when focused programmatically; individual control focus indicators remain.

## Dependency remediation

The initial upgrade had 79 inherited audit findings. Dependency overrides now update the affected CRA/Jest/SVG/HTTP transitive packages while retaining the existing application framework. The development startup adapter uses webpack-dev-server 5, its current middleware interface and host validation. It removes CRA's blanket cross-origin development headers. Optional `CLASSIO_DEV_ALLOWED_HOSTS` accepts a comma-separated list of explicitly trusted development hostnames; the default is `auto`.

The registry's latest picomatch release, 4.0.7, remained vulnerable to CVE-2026-112366 / GHSA-fc5q-39xc-vjpf. Classio therefore includes a **local security fork**, whose package version is 4.0.8 to satisfy existing optional peer constraints. This is not a claim that upstream published 4.0.8. The patch prevents the reported single-branch repeated extglobs from compiling into nested unbounded regular expressions, following upstream's existing literalization behavior for unsafe syntax. No advisory is excluded from the audit.

Reviewable source, the original MIT license, upstream integrity, local archive/source hashes and maintenance instructions are in [vendor/picomatch](../vendor/picomatch/SECURITY-PATCH.md). Installation uses the committed tarball with a root override. Vendor integrity checks compare source, archive and installed bytes. Isolated timeout tests exercise the reported attack patterns, including disabled optional recursion limits, and verify ordinary matching and actual test-discovery patterns. Replace the fork when an upstream fix passes these checks.

## Verification

- A fresh `npm ci --ignore-scripts` succeeds against the committed lockfile and local archive. `npm audit --audit-level=low` reports **zero findings at every severity**, including development dependencies, on this date.
- **226 frontend tests across 50 suites** and **67 server tests** pass. Production build passes, retaining existing lint/bundle-size warnings.
- Real Firebase Admin verification uses temporary, runtime-generated RSA keys in tests. Valid signatures authorize; expired, forged and wrong-project tokens fail. Certificate outages and invalid administrator/emulator setup report service/configuration errors. The test replaces Google's certificate transport, not JWT verification.
- `npm run test:auth-browser` runs actual `/api/session` and `/api/study` route handlers over HTTP in Chromium. It verifies token refresh, secure HttpOnly guest cookies, CSRF renewal, an unchanged payload, bounded retries, one inference per successful request, and safe missing-configuration errors. Upstream inference and Google certificate transport are explicit fixtures; this is not a live-provider claim.
- Responsive browser checks pass at desktop, both tablet orientations and phone sizes, including the compact empty-attempt message and a 20-slide presentation interrupted, refreshed and resumed without duplicate slides. Tablet presentation editing/export checks also pass.
- The upgraded development server compiles and returns Classio HTML. A raw HTTP request with an untrusted Host header returns 403.
- Source, staged changes, available Git history and the production bundle pass private-credential scanning; scanner canaries also pass. The existing public Firebase web configuration remains an exact-hash reviewed public configuration finding.

## Vercel verification still required

There is no Vercel administrative binding or CLI connection in this cloud runtime, and its network proxy rejects access to `classio-eta.vercel.app`. Consequently the screenshots do not establish which production setting caused the failure, and this report does not claim a healthy live deployment or verified live AI response.

No Vercel variables were read, changed or removed. Their exact names, including `packup`, remain preserved. Required production settings remain `CLASSIO_SESSION_SECRET` (at least 32 characters), `AI_QUOTA_MODE=firestore`, a valid `FIREBASE_SERVICE_ACCOUNT_JSON` for the application's Firebase project, and at least one permitted provider key plus its student-use/free-tier confirmations. `FIREBASE_PROJECT_ID`, if set, must match that project. Actual Vercel presence is **unknown**, rather than confirmed missing. Run `npm run check:ai` in the server environment and inspect the fixed `study_access` / `study_configuration` Vercel log diagnostics to identify the specific missing setting without exposing values.

Gemini Developer API remains blocked for Classio's intended student audience. Other providers retain their eligibility and free-use gates. No credential is embedded in source or public build variables. The TTS repository is unchanged.
