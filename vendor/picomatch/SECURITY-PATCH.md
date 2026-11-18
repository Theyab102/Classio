# Local picomatch security fork

This is **Classio's private local fork**, not an upstream 4.0.8 release. Upstream
4.0.7 was the latest published version when this change was made. The local
package uses version 4.0.8 so existing `^4` optional peer requirements resolve;
the local tarball, override, source and integrity manifest identify this fork.

Source: the npm `picomatch@4.0.7` distribution, with its original MIT license.
The upstream tarball integrity was verified by npm when downloaded. Its SRI is
recorded in `integrity.json`. No source from an unverified binary is used.

CVE-2026-112366 / GHSA-fc5q-39xc-vjpf reports exponential backtracking for
single-branch repeated extglobs such as `+(*)` and `+(*a)`. The upstream
`analyzeRepeatedExtglob` guard only applied its wildcard test to multiple
branches. This fork adds a mandatory check for unescaped unbounded wildcards or
nested groups in a single repeated branch. Unsafe shapes follow the existing
upstream guard's behavior: they are literalized instead of compiled to nested,
unbounded regular expressions. Escaped characters and bracket classes remain
recognized. Safe ordinary globs, literal repeats and multibranch test globs stay
supported. The guard also applies when optional recursion limits are disabled.
This deliberately restricts unsafe glob syntax; it is not a claim that arbitrary
regular-expression extensions can be safely accepted from untrusted users.

`server/glob-security.test.js` executes the reported patterns in a separate
process with a hard deadline and checks ordinary matching/test discovery.
The application suites ensure that the override remains compatible with the
existing build, test and development tools. No audit exclusion is configured.
`check-security-vendor.cjs` verifies the reviewed source and tarball bytes.

Remove this fork once an upstream fix covering these cases is released and
passes the regression, application and audit checks. Do not simply relabel an
unpatched package or remove the regression tests to make an audit pass.

References:
- https://github.com/advisories/GHSA-fc5q-39xc-vjpf
- https://github.com/micromatch/picomatch/issues/194
