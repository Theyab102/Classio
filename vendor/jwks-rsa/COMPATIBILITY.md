# jwks-rsa 4.1.0 CommonJS compatibility patch

Production returned support reference A102 (`ERR_REQUIRE_ESM`). The same error was reproduced locally using Node 24 with `--no-experimental-require-module`: Firebase Admin imports jwks-rsa, whose `src/utils.js` and `src/integrations/passport.js` synchronously require the ES-module-only jose 6 package. Importing Firebase Admin dynamically does not repair a transitive CommonJS require.

This derivative changes those two loading sites to native `import('jose')`. The signing-key converter already returns a promise. The Passport integration retains its callback signature and undefined return value. Key filtering, algorithms, signature verification, HTTP retrieval, caching and rate limiting retain upstream behavior. The currently maintained jose 6 dependency is retained; it is not replaced with an older cryptography release, and no runtime verification flags are weakened.

Source: npm `jwks-rsa@4.1.0`, https://github.com/auth0/node-jwks-rsa. The original tarball was checked against its registry/lockfile SHA-512 integrity before extraction. The upstream MIT license and distributed files are retained. The derivative version is explicitly `4.1.0-classio.2`; it is not represented as an upstream release.

`integrity.json` records the original npm integrity, reviewed tarball SHA-256 and distributed source hashes. `scripts/check-security-vendor.cjs` verifies the archive, reviewed sources and installed package. `server/firebase-cjs.test.js` runs real Firebase JWT verification, forged/expired/wrong-project rejection and actual JWKS/Passport integration tests in a child Node process with synchronous require(ESM) disabled.
