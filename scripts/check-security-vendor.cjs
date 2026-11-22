const fs = require("node:fs"),
  path = require("node:path"),
  crypto = require("node:crypto");
const root = path.resolve(__dirname, ".."),
  manifests = [
    "vendor/picomatch/integrity.json",
    "vendor/jwks-rsa/integrity.json",
  ];
const hash = (file) =>
  crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
for (const location of manifests) {
  const manifest = require(path.join(root, location)),
    base = path.dirname(path.join(root, location));
  if (hash(path.join(root, manifest.archive)) !== manifest.archiveSha256)
    throw Error("Reviewed archive integrity failed: " + location);
  const installed = path.dirname(
    require.resolve((manifest.package || "picomatch") + "/package.json"),
  );
  const files = Array.isArray(manifest.files)
    ? manifest.files.map(({ path: file, sha256 }) => [file, sha256])
    : Object.entries(manifest.files);
  for (const [file, expected] of files) {
    if (
      hash(path.join(base, file)) !== expected ||
      hash(path.join(installed, file)) !== expected
    )
      throw Error(
        "Reviewed source/installed integrity failed: " + location + "/" + file,
      );
  }
}
