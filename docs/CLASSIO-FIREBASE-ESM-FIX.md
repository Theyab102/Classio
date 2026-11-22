# Fix for production authentication reference A102

The owner's live retry after the diagnostic deployment returned A102. This identifies `ERR_REQUIRE_ESM`, before provider generation. The failure was reproduced locally with Node 24 and `--no-experimental-require-module`: Firebase Admin 14.5.0 imports jwks-rsa 4.1.0, which synchronously requires jose 6.2.15. A dynamic import of Firebase Admin itself still fails inside that dependency.

The reviewed `jwks-rsa@4.1.0-classio.2` derivative changes only the two jose loading sites to native dynamic imports. It retains the current jose 6 cryptography implementation and all existing key/signature/revocation checks. Upstream's published 4.1.0 still had synchronous requires when inspected. No SDK verification, TLS or package integrity check is disabled, and no provider eligibility or billing flags are changed.

The root package and override resolve Firebase Admin to this reviewed tarball. The upstream archive was verified against the existing lockfile integrity, and the patch/archive/installed bytes are checked by the existing vendor integrity gate. See `vendor/jwks-rsa/COMPATIBILITY.md` and its integrity manifest.

The regression suite exercises real Firebase signatures, rejection of forged/expired/wrong-project tokens, real public-JWK conversion, rejection of private key material and callback integration. These checks also run under the exact Node loader restriction that reproduced the production failure.

After deployment, retry the original signed-in generation request. Resolving A102 proves that this loader failure is gone; it does not establish that provider credentials, production quotas or Firestore permissions are correctly configured. Inspect any subsequent error separately and preserve existing user data and free-only operation.
