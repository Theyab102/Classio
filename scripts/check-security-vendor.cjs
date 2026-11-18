const fs = require("node:fs"),
  path = require("node:path"),
  crypto = require("node:crypto");
const root = path.resolve(__dirname, ".."),
  manifest = require("../vendor/picomatch/integrity.json"),
  base = path.join(root, "vendor/picomatch");
const hash = (file) =>
  crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
if (hash(path.join(root, manifest.archive)) !== manifest.archiveSha256)
  throw Error("Security fork archive integrity failed.");
const installed = path.dirname(require.resolve("picomatch"));
for (const [file, expected] of Object.entries(manifest.files)) {
  if (
    hash(path.join(base, file)) !== expected ||
    hash(path.join(installed, file)) !== expected
  )
    throw Error("Security fork source/installed integrity failed: " + file);
}
