const { test } = require("node:test"),
  assert = require("node:assert/strict");
const fs = require("node:fs"),
  path = require("node:path");
const { reviewedFirebaseConfig } = require("../scripts/reviewed-firebase.cjs");
const source = fs.readFileSync(path.join(__dirname, "../src/App.js"), "utf8");
const publicKey = /apiKey:\s*["'](AIza[\w-]+)["']/.exec(source)[1];
const context = 'initializeApp({authDomain:"classio-4378f.firebaseapp.com"});';
test("the exact reviewed public browser config supports literal and environment-fallback syntax at reviewed paths", () => {
  for (const file of ["src/App.js", "src/firebase.js"])
    for (const line of [
      `apiKey: "${publicKey}"`,
      `apiKey: process.env.REACT_APP_FIREBASE_API_KEY || "${publicKey}"`,
    ])
      assert(reviewedFirebaseConfig(file, context, line));
});
test("different keys, source locations, fields and projects remain blocked", () => {
  const line = `apiKey: "${publicKey}"`;
  assert(!reviewedFirebaseConfig("src/other.js", context, line));
  assert(
    !reviewedFirebaseConfig(
      "src/firebase.js",
      context,
      `token: "${publicKey}"`,
    ),
  );
  assert(
    !reviewedFirebaseConfig(
      "src/firebase.js",
      context,
      line.replace(publicKey, "AIza" + "X".repeat(35)),
    ),
  );
  assert(
    !reviewedFirebaseConfig(
      "src/firebase.js",
      context.replace("classio-4378f", "another-project"),
      line,
    ),
  );
  assert(
    !reviewedFirebaseConfig(
      "src/firebase.js",
      context.replace("initializeApp", "unrelated"),
      line,
    ),
  );
  assert(
    !reviewedFirebaseConfig(
      "src/firebase.js",
      context,
      line + `, another: "AIza${"X".repeat(35)}"`,
    ),
  );
});
