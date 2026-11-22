const { createHash } = require("node:crypto");
const publicFirebaseHash =
  "f44974192bbe24af103ac60856d08e20f1dcc9de8b0aae2a205644596555c596";
function reviewedFirebaseConfig(file, content, line) {
  if (!["src/App.js", "src/firebase.js"].includes(file.replaceAll("\\", "/")))
    return false;
  if (!/\bapiKey\s*:/.test(line)) return false;
  const keys = line.match(/AIza[\w-]{30,50}/g) || [];
  return (
    keys.length === 1 &&
    createHash("sha256").update(keys[0]).digest("hex") === publicFirebaseHash &&
    content.includes("classio-4378f.firebaseapp.com") &&
    content.includes("initializeApp")
  );
}
module.exports = { publicFirebaseHash, reviewedFirebaseConfig };
