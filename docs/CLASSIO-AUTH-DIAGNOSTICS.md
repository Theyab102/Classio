# Authentication failure diagnostics

The production screenshot's generic `AUTH_UNAVAILABLE` does not identify why Firebase verification failed. This update adds allowlisted log classifications and a short public support reference. It does not bypass authentication or establish that production generation is fixed.

`study_access` logs contain only a public code, fixed reason, fixed verification phase and support reference. Raw SDK messages, JWTs, administrator JSON, keys and internal paths are excluded. Public responses keep the same `code`/`error` fields and general user message, with a short `Reference: A…` suffix for service/configuration failures.

| Reference | Operator diagnosis |
| --- | --- |
| A100–A104 | SDK/module loading; A101 missing module, A102 ESM require, A103 asynchronous ESM require, A104 package export |
| A201 | Firebase public certificate lookup |
| A202 | Firebase verification transport/network |
| A300–A303 | Server configuration; A302 project mismatch, A303 invalid administrator credential format |
| A401 | Firebase SDK internal error |
| A408 | Verification deadline |
| A500 | Unclassified verification failure; inspect the sanitized phase in Vercel logs |

Project, issuer, signature, Google-provider, revocation, origin and emulator checks retain their existing behavior. Expired/revoked accounts remain unauthorized and never become guest identities. Provider activation, quotas, model routing, image services, stored user data and the editor are unchanged.

After this commit is deployed, retry the failing request once. The error reference is sufficient to identify the initial failure category; the corresponding `study_access` entry supplies the phase. Do not include credentials, authorization headers or full tokens in screenshots or reports.

Production provider configuration and live generation still require separate verification. The full routing/image changes remain on `fix/ai-routing-and-recovery` until their configuration can be validated.

The full-history credential check also encountered an inherited false positive in `src/firebase.js` on an older remote branch: its Firebase web key has the same already reviewed fingerprint as the current browser config, but uses an environment-variable fallback. The review now permits only that exact fingerprint in the `apiKey` field at the two known browser config paths, with the existing Classio Firebase domain and SDK initialization present. Other values, files and fields remain blocking; no history is deleted, and scanner canaries still exercise private credentials.
