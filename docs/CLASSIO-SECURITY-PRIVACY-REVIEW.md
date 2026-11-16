# Security and privacy review

This records code-level checks and fixes on 16 November 2026. It is not a claim
that every vulnerability or legal obligation has been eliminated. Social-video
fine amounts were not treated as verified legal facts.

## The twenty threat examples

| Example | Classio finding and action |
| --- | --- |
| Skip login with a header | Guest study access is intentional. Protected sharing verifies a revoked-checked Firebase token and Google provider; browser headers are not treated as identity. Existing auth/quota tests remain passing. |
| Download `.git` / environment files | Production serves the generated build, with explicit private-path 404 rewrites. No checkout or secret files were added to public assets. |
| Read public storage | Owner-only Firestore source rules remain. Added deny-by-default Storage rules for the current device-local upload architecture. Neither rule set was deployed or substituted for unknown live rules. Production permissions remain an operator verification. |
| Change checkout prices | No checkout or subscription system exists; none was added. |
| Credential stuffing | Classio has Google sign-in, not an application password endpoint. Google account protections remain with that provider. |
| Wipe database | Client rules restrict account paths to the Google-authenticated owner. Cross-account/restore regressions remain. This does not prove deployed rules match source. |
| Compromise a CDN script | Mammoth, Mermaid, Chart and the legacy spreadsheet reader are verified local assets with license/provenance records. Main-thread document loaders use SRI; builds verify hashes. CSP restricts scripts to self and Firebase's required Google auth loader. |
| Run up SMS charges | No SMS feature exists. Public speech generation now has bounded text/body size, concurrent work and per-instance rate limits. |
| Make chat expose instructions | Provider credentials are never placed in model prompts or browser bundles; the assistant has no account/file mutation tools. Source text is untrusted context. System instructions are not a secret-storage boundary. |
| Forge JWT using an exposed secret | No application signing key is shipped. Protected operations verify Firebase ID tokens on the server. |
| Reuse coupons | No coupon system exists. |
| Pre-register another person's email | No application email/password registration exists; Google identity is used. |
| Forgotten subdomain takeover | Requires the operator's actual DNS/hosting inventory. No custom-domain administration is available here, so this remains external. |
| SSRF / cloud metadata | Public-page reader validates public IPs, pins DNS, and revalidates redirects. Transcript fetches reject redirects rather than following a validated URL to another host. Existing private-IP/reader tests pass. |
| Filesystem traversal | No arbitrary server file-reading API is introduced. Voice IDs are allowlisted, and private-path deployment rewrites are explicit. Document archives have size/entry/expanded-size limits. |
| Poison password reset host | No custom password-reset endpoint exists. |
| Read source maps | Production maps are disabled and the build contains zero `.map` files. Browser-delivered JavaScript is still public; secrets must remain server-side. |
| Invisible iframe / clickjacking | Retained `X-Frame-Options: DENY`, added CSP `frame-ancestors 'none'`, plus no-sniff/base/object/form restrictions. Deployment activation remains unverified. |
| Open redirect to phishing | No user-controlled navigation redirect endpoint exists. Policy rewrites are fixed; fetched webpages and credit URLs do not become arbitrary server redirects. |
| Enumerate account emails | No Classio email/account lookup endpoint exists. This does not audit Google's own service. |

## Additional fixes and checks

- Remove.bg requests check actual format signatures and byte limits before
  making a paid request. Output must be PNG. Credential, credit, rate and timeout
  failures have useful messages without returning provider bodies/secrets.
- Generated/uploaded illustrations are staged. Aborted or stale requests cannot
  overwrite newer edits. Animated image removal is not silently attempted.
- Production uses CSP, HSTS, referrer restrictions, no-sniff and a restricted
  permissions policy. Google auth popups retain the required opener behavior.
- The XLS/XLSX parser runs in a dedicated worker with a time limit, response
  bounds and no permitted network access under its deployment CSP. It cannot
  access the main window DOM. The older parser still needs a maintained upstream
  replacement; isolation is mitigation, not a claim that its code is patched.
- Office archives reject excessive expanded data before extraction. PDF text
  extraction is bounded. Image filenames no longer prompt invented source text.
- Math export bounds macro expansion and rejects unsupported link/HTML/loader
  commands before producing vector output.
- Material deletion attempts local original/audio cleanup, and failures are
  reported rather than silently described as complete. Cloud writes retain
  their pending/local cache and visible failure notice. Large workspaces remain
  subject to Firebase document and browser-storage limits.
- The speech service caps requests and concurrent generation, limits per-client
  and total instance traffic, validates speed/text, avoids public caching of
  personal audio and omits internal exception details from responses. Failed
  downloads cannot leave a half-published model/config pair. Its five API tests
  use mocked synthesis; actual model generation is not claimed.
- A built-file scan found no matches for the checked provider/private-key
  patterns. This is a bounded pattern scan, not proof that every possible secret
  format is absent. Existing public Firebase client configuration is retained.

## Dependencies

Compatible lockfile updates and targeted overrides reduce `npm audit` findings
from **63 (20 high)** to **34 (6 high)**, with zero reported critical findings in
the final inventory. Installed overrides include patched gRPC, image-size,
PostCSS, serialization and supporting packages; the actual build and tests pass.

The six remaining high findings are in the legacy CRA build/development chain:
`@svgr/plugin-svgo`, `css-select`, `nth-check`, `svgo`,
`webpack-dev-middleware`, and `webpack-dev-server`. Remaining moderate/low
findings also remain in the inventory. They were not hidden by forcing an
incompatible upgrade or removing audit coverage. Do not expose the development
server publicly; production uses static output and serverless API functions.
Migrating the legacy toolchain needs a dedicated, validated change.

Local document assets are recorded in `public/vendor/manifest.json` with package
versions and registry integrity hashes. Their loader hashes are in
`src/localScriptManifest.json`; `scripts/check-vendor.cjs` enforces them before
build. Replacing these files requires reviewing the source and updating the
matching manifests, not merely changing an expected hash to suppress a failure.

## Privacy and legal applicability

The privacy policy now describes actual presentation generation, remove.bg,
saved avatar assets, local file/audio deletion, cloud-sync failures and provider
retention limits. The withdrawn sketch creator is not presented as live.
Terms clarify free access and that Classio is not an emergency/clinical service.
A narrow English/Arabic local supportive response handles explicit self-harm
disclosures even without an AI provider; additional server instructions guide
configured providers. It is not a clinical classifier or a guarantee of crisis
detection in every language or wording.

Self-hosted fonts and the absence of app analytics/session replay are preserved.
The age/terms screen remains a self-declared 13+ check, not verified identity or a
guarantee of regulatory compliance. Gemini retains explicit opt-in. Upload rights
notices and copyright reporting pages remain. No fake testimonials, subscriptions,
checkout, renewal or marketing-email flow was introduced.

The operator must still supply real monitored contact details and legal identity,
review applicable jurisdiction/provider terms, and make any required rights-agent
registration. The app cannot invent those facts, erase provider/hosting logs or
guarantee deletion from other devices by clearing this browser. Subscription and
trial disclosures do not apply to the current free product.

## External actions not performed

No Vercel deployment, production secret update, Firebase rule deployment,
provider spending-limit change, DNS audit or legal filing was performed. Instance
quotas reset and are not a deployment-wide hard spend cap. Production provider
limits and, when appropriate, shared quotas still need operator configuration.
The production URL and live voice-model hosts are blocked by the current cloud
network policy. No live remove.bg, Groq, Gemini, Firebase login or TTS success is
inferred from fixture tests. Physical Safari/iPad and desktop PowerPoint checks
remain external device/application validation.
